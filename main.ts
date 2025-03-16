import { App, Editor, MarkdownView, Modal, Notice, Plugin, PluginSettingTab, Setting, TFile } from 'obsidian';
import * as fs from 'fs';
import * as path from 'path';

// -------------------------------------------------------------------
// Interfaces
// -------------------------------------------------------------------

interface OllamaSearchPluginSettings {
	ollamaEndpoint: string;
	modelName: string;
	maxResults: number;
	chunkSize: number;        // New: chunk size in characters
	chunkOverlap: number;     // New: overlap in characters
	errorLogs: ErrorLogEntry[]; // New: store error logs
}

interface VectorIndexEntry {
	path: string;
	title: string;
	mtime: number;         // Last modified time
	chunkIndex: number;    // Which chunk number of the file
	content: string;       // The chunk text
}

interface VectorIndex {
	[id: string]: VectorIndexEntry;
}

interface SearchResult {
	id: string;
	similarity: number;
	path: string;
	title: string;
	content: string; // The chunk content for this search result
}

interface ErrorLogEntry {
	timestamp: number;
	message: string;
	details?: string;
}

const DEFAULT_SETTINGS: OllamaSearchPluginSettings = {
	ollamaEndpoint: 'http://localhost:11434',
	modelName: 'jeffh/intfloat-multilingual-e5-large-instruct:q8_0',
	maxResults: 5,
	chunkSize: 1000,      // Default chunk size in characters
	chunkOverlap: 200,    // Default overlap in characters
	errorLogs: []         // Initialize empty error logs
};

// -------------------------------------------------------------------
// Main Plugin
// -------------------------------------------------------------------

export default class OllamaSearchPlugin extends Plugin {
	settings: OllamaSearchPluginSettings;
	vectorsPath: string;
	indexPath: string;
	vectorIndex: VectorIndex = {};
	isIndexed: boolean = false;
	isIndexing: boolean = false; // FIX: Add flag to track indexing state
	private vectorsCache: Map<string, number[]> = new Map();
	private forceReindex: boolean = false;

	async onload() {
		await this.loadSettings();

		// Set up file paths
		const basePath = this.manifest.dir || '';
		this.vectorsPath = path.join(basePath, 'vectors.bin');
		this.indexPath = path.join(basePath, 'vector_index.json');

		// Check if index exists
		this.isIndexed = await this.checkIfIndexExists();

		// Add ribbon icon for search
		this.addRibbonIcon('search', 'Ollama Search', (evt: MouseEvent) => {
			if (!this.isIndexed) {
				new Notice('Please build the vector index first in settings');
				return;
			}
			if (this.isIndexing) {
				new Notice('Indexing in progress. Please wait...');
				return;
			}
			new SearchModal(this.app, this).open();
		});

		// Add command for search
		this.addCommand({
			id: 'open-ollama-search',
			name: 'Open Ollama Search',
			callback: () => {
				if (!this.isIndexed) {
					new Notice('Please build the vector index first in settings');
					return;
				}
				if (this.isIndexing) {
					new Notice('Indexing in progress. Please wait...');
					return;
				}
				new SearchModal(this.app, this).open();
			}
		});

		// Add settings tab
		this.addSettingTab(new OllamaSettingTab(this.app, this));

		// Load index and vectors if they exist
		if (this.isIndexed) {
			try {
				await this.loadVectorIndex();
				try {
					this.vectorsCache = await this.loadVectorsFromBinary();
					// If we got an empty cache but the index exists, something is wrong
					if (this.vectorsCache.size === 0 && Object.keys(this.vectorIndex).length > 0) {
						this.logError("Vector data appears to be corrupted or empty", "Binary file could not be read properly");
						new Notice("Vector data appears to be corrupted. Please rebuild the index.");
						this.isIndexed = false;
					}
				} catch (error) {
					this.logError("Failed to load vectors from binary file", error.message);
					new Notice("Vector data appears to be corrupted. Please rebuild the index.");
					this.isIndexed = false;
				}
			} catch (error) {
				this.logError("Failed to load vector index", error.message);
				new Notice("Vector index appears to be corrupted. Please rebuild the index.");
				this.isIndexed = false;
			}
		}
	}

	// -------------------------------------------------------------------
	// Index existence check
	// -------------------------------------------------------------------
	async checkIfIndexExists(): Promise<boolean> {
		try {
			const adapter = this.app.vault.adapter;
			const vectorsExist = await adapter.exists(this.vectorsPath);
			const indexExists = await adapter.exists(this.indexPath);
			return vectorsExist && indexExists;
		} catch (error) {
			console.error("Error checking index:", error);
			return false;
		}
	}

	// -------------------------------------------------------------------
	// Load and save index
	// -------------------------------------------------------------------
	async loadVectorIndex() {
		try {
			const adapter = this.app.vault.adapter;
			const indexData = await adapter.read(this.indexPath);
			this.vectorIndex = JSON.parse(indexData);
			console.log(`Loaded vector index with ${Object.keys(this.vectorIndex).length} entries`);
		} catch (error) {
			console.error("Error loading vector index:", error);
			this.vectorIndex = {};
		}
	}

	async saveVectorIndex() {
		try {
			const adapter = this.app.vault.adapter;
			await adapter.write(this.indexPath, JSON.stringify(this.vectorIndex, null, 2));
			console.log(`Saved vector index with ${Object.keys(this.vectorIndex).length} entries`);
		} catch (error) {
			console.error("Error saving vector index:", error);
			throw error;
		}
	}

	// -------------------------------------------------------------------
	// Building / Rebuilding the index
	// -------------------------------------------------------------------
	async buildVectorIndex(forceRebuild: boolean = false) {
		// Set the force reindex flag
		this.forceReindex = forceRebuild;
		this.isIndexing = true; // FIX: Set indexing flag
		
		const files = this.app.vault.getMarkdownFiles();
		let indexingNotice = new Notice(`Indexing 0 / ${files.length} notes...`, 0);

		// If forcing a full rebuild, clear the existing index first
		if (this.forceReindex) {
			this.vectorIndex = {};
			this.vectorsCache.clear();
		}

		try { // FIX: Wrap in try/catch to ensure isIndexing is reset
			for (const [i, file] of files.entries()) {
				const stat = await this.app.vault.adapter.stat(file.path);
				if (!stat) {
					// Can't retrieve file stats; skip
					continue;
				}

				// Check if this file is already indexed and up to date
				const oldChunks = Object.entries(this.vectorIndex)
					.filter(([id, entry]) => entry.path === file.path);

				const isUpToDate = !this.forceReindex && 
					oldChunks.length > 0 && 
					oldChunks.every(([id, entry]) => entry.mtime === stat.mtime);
					
				if (isUpToDate) {
					continue;
				}

				// -- Minimal fix #1: Remove stale chunks for this file before reindexing
				for (const [oldId, entry] of oldChunks) {
					delete this.vectorIndex[oldId];
					this.vectorsCache.delete(oldId);
				}

				try {
					const content = await this.app.vault.read(file);
					const chunks = this.chunkText(
						content,
						this.settings.chunkSize,
						this.settings.chunkOverlap
					);

					// Generate embeddings per chunk
					for (let cIndex = 0; cIndex < chunks.length; cIndex++) {
						const chunk = chunks[cIndex];
						const embedding = await this.getEmbedding(chunk);

						// -- Minimal fix #3: Skip storing chunk if embedding is empty
						if (embedding.length === 0) {
							console.warn(`Embedding for ${file.path} chunk ${cIndex} is empty; skipping.`);
							continue;
						}

						// -- Minimal fix #4: Ensure unique ID by checking for collisions
						let id: string;
						do {
							id = this.generateId();
						} while (this.vectorIndex[id]);

						// Store in memory
						this.vectorsCache.set(id, embedding);

						// Update index
						this.vectorIndex[id] = {
							path: file.path,
							title: file.basename,
							mtime: stat.mtime,
							chunkIndex: cIndex,
							content: chunk
						};
					}

					if ((i + 1) % 10 === 0) {
						indexingNotice.setMessage(`Indexing ${i + 1} / ${files.length} notes...`);
					}
				} catch (error) {
					console.error(`Error indexing ${file.path}:`, error);
				}
			}

			// Save vectors to binary file
			await this.saveVectorsToBinary(
				Array.from(this.vectorsCache.entries()).map(([id, vector]) => ({ id, vector }))
			);

			// Save index
			await this.saveVectorIndex();
			this.isIndexed = true;
		} catch (error) {
			this.logError("Error during indexing", error.message);
			throw error;
		} finally {
			indexingNotice.hide();
			new Notice('Indexing complete!');
			this.isIndexing = false; // FIX: Reset indexing flag
		}
	}

	// -------------------------------------------------------------------
	// Text chunking logic
	// -------------------------------------------------------------------
	private chunkText(text: string, size: number, overlap: number): string[] {
		// -- Minimal fix #2: Clamp overlap to avoid infinite loops
		overlap = Math.min(overlap, size - 1);

		const chunks: string[] = [];
		
		// FIX: Improve chunking to respect natural boundaries
		// Split text into paragraphs first
		const paragraphs = text.split(/\n\s*\n/);
		let currentChunk = "";
		
		for (const paragraph of paragraphs) {
			// If adding this paragraph would exceed the chunk size, 
			// save the current chunk and start a new one
			if (currentChunk.length + paragraph.length > size && currentChunk.length > 0) {
				chunks.push(currentChunk);
				// Include overlap from the end of the previous chunk
				const overlapText = currentChunk.length > overlap 
					? currentChunk.slice(-overlap) 
					: currentChunk;
				currentChunk = overlapText + paragraph;
			} else {
				// Add paragraph to current chunk
				if (currentChunk.length > 0) {
					currentChunk += "\n\n";
				}
				currentChunk += paragraph;
			}
			
			// If current chunk exceeds size, split it further
			while (currentChunk.length > size) {
				const chunkToAdd = currentChunk.slice(0, size);
				chunks.push(chunkToAdd);
				currentChunk = currentChunk.slice(size - overlap);
			}
		}
		
		// Add the last chunk if it's not empty
		if (currentChunk.length > 0) {
			chunks.push(currentChunk);
		}
		
		return chunks;
	}

	// -------------------------------------------------------------------
	// Random ID generator (for chunk entries) -- slightly updated
	// -------------------------------------------------------------------
	generateId(): string {
		// We'll leave the generation logic but let the caller verify collisions.
		return Math.random().toString(36).substring(2, 15) +
			   Math.random().toString(36).substring(2, 15);
	}

	// -------------------------------------------------------------------
	// Binary read & write for vector data
	// -------------------------------------------------------------------
	private alignOffset(view: DataView, offset: number): number {
		while (offset % 4 !== 0) {
			view.setUint8(offset, 0);
			offset++;
		}
		return offset;
	}

	async saveVectorsToBinary(vectors: { id: string, vector: number[] }[]): Promise<void> {
		try {
			// Format: For each vector:
			//   [id_length (2 bytes)][id (variable)]
			//   [align to multiple of 4 with 0s]
			//   [vector_length (4 bytes)][vector_data (float32 array)]
			const adapter = this.app.vault.adapter;

			// Calculate total size
			let totalSize = 0;
			for (const item of vectors) {
				totalSize += 2; // id length (2 bytes)
				totalSize += item.id.length; // id string
				// We'll align after ID, so add up to 3 possible padding bytes
				totalSize += 3;
				totalSize += 4; // vector length (4 bytes)
				totalSize += item.vector.length * 4; // float32 array
			}

			const buffer = new ArrayBuffer(totalSize);
			const view = new DataView(buffer);

			let offset = 0;
			for (const item of vectors) {
				// Write id length (2 bytes)
				view.setUint16(offset, item.id.length, true);
				offset += 2;

				// Write id string
				for (let i = 0; i < item.id.length; i++) {
					view.setUint8(offset + i, item.id.charCodeAt(i));
				}
				offset += item.id.length;

				// Align to multiple of 4 before writing vector length
				offset = this.alignOffset(view, offset);

				// Write vector length (4 bytes)
				view.setUint32(offset, item.vector.length, true);
				offset += 4;

				// Write vector data
				for (let i = 0; i < item.vector.length; i++) {
					view.setFloat32(offset, item.vector[i], true);
					offset += 4;
				}
			}

			const uint8Array = new Uint8Array(buffer);
			await adapter.writeBinary(this.vectorsPath, uint8Array);

			console.log(`Saved ${vectors.length} vectors to binary file`);
		} catch (error) {
			console.error("Error saving vectors to binary:", error);
			throw error;
		}
	}

	async loadVectorsFromBinary(): Promise<Map<string, number[]>> {
		try {
			const adapter = this.app.vault.adapter;
			const arrayBuffer = await adapter.readBinary(this.vectorsPath);

			const vectors = new Map<string, number[]>();
			const view = new DataView(arrayBuffer);

			let offset = 0;
			// Add a safety check to prevent infinite loops
			let safetyCounter = 0;
			const maxIterations = 100000; // Reasonable upper limit
			
			while (offset < arrayBuffer.byteLength && safetyCounter < maxIterations) {
				safetyCounter++;
				
				try {
					// Check if we have enough bytes left to read the ID length (2 bytes)
					if (offset + 2 > arrayBuffer.byteLength) {
						console.warn("Reached end of buffer while reading ID length");
						break;
					}
					
					// Read id length
					const idLength = view.getUint16(offset, true);
					offset += 2;

					// Validate ID length to prevent buffer overruns
					if (idLength <= 0 || idLength > 100 || offset + idLength > arrayBuffer.byteLength) {
						console.warn(`Invalid ID length: ${idLength} at offset ${offset-2}`);
						// Skip to next potential valid entry
						offset = Math.min(offset + 4, arrayBuffer.byteLength);
						continue;
					}

					// Read id
					let id = '';
					for (let i = 0; i < idLength; i++) {
						id += String.fromCharCode(view.getUint8(offset + i));
					}
					offset += idLength;

					// Align to multiple of 4
					while (offset % 4 !== 0 && offset < arrayBuffer.byteLength) {
						offset++;
					}

					// Check if we have enough bytes left to read the vector length (4 bytes)
					if (offset + 4 > arrayBuffer.byteLength) {
						console.warn("Reached end of buffer while reading vector length");
						break;
					}

					// Read vector length
					const vectorLength = view.getUint32(offset, true);
					offset += 4;

					// Validate vector length to prevent buffer overruns
					if (vectorLength <= 0 || vectorLength > 10000 || offset + (vectorLength * 4) > arrayBuffer.byteLength) {
						console.warn(`Invalid vector length: ${vectorLength} at offset ${offset-4}`);
						// Skip to next potential valid entry
						offset = Math.min(offset + 4, arrayBuffer.byteLength);
						continue;
					}

					// Read float32 array
					const vector = new Array(vectorLength);
					for (let i = 0; i < vectorLength; i++) {
						vector[i] = view.getFloat32(offset, true);
						offset += 4;
					}

					vectors.set(id, vector);
				} catch (error) {
					// Log error but try to continue with next vector
					console.error("Error reading vector at offset", offset, error);
					// Skip ahead by 4 bytes to try to recover
					offset = Math.min(offset + 4, arrayBuffer.byteLength);
				}
			}
			
			if (safetyCounter >= maxIterations) {
				console.warn("Reached maximum iterations while reading vectors file. File may be corrupted.");
			}

			console.log(`Loaded ${vectors.size} vectors from binary file`);
			return vectors;
		} catch (error) {
			console.error("Error loading vectors from binary:", error);
			// Return empty map instead of throwing to prevent crashes
			return new Map<string, number[]>();
		}
	}

	// -------------------------------------------------------------------
	// Embedding & Similarity
	// -------------------------------------------------------------------
	async getEmbedding(text: string, retryCount = 0): Promise<number[]> {
		const maxRetries = 2; // Allow up to 2 retries
		
		try {
			const response = await fetch(`${this.settings.ollamaEndpoint}/api/embed`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: this.settings.modelName,
					input: text
				})
			});
			if (!response.ok) {
				const errorText = await response.text();
				throw new Error(`Ollama responded with ${response.status} ${response.statusText}: ${errorText}`);
			}
			const data = await response.json();
			
			// The response format is different - embeddings is an array of arrays
			// We want the first (and likely only) embedding
			return data.embeddings?.[0] || [];
		} catch (err) {
			this.logError(`Failed to get embedding from Ollama`, err.message || err.toString());
			
			new Notice(`Failed to get embedding from Ollama: ${err.message}`);
			console.error(err);
			// Return an empty embedding to avoid further errors
			return [];
		}
	}

	cosineSimilarity(vecA: number[], vecB: number[]) {
		// Basic guard for length mismatch
		if (vecA.length !== vecB.length || vecA.length === 0) {
			return 0;
		}
		const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
		const magA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
		const magB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
		
		// FIX: Handle zero magnitude vectors
		if (magA === 0 || magB === 0) {
			return 0;
		}
		
		return dotProduct / (magA * magB);
	}

	// -------------------------------------------------------------------
	// Main search routine
	// -------------------------------------------------------------------
	async searchSimilar(query: string): Promise<SearchResult[]> {
		try {
			// FIX: Check if indexing is in progress
			if (this.isIndexing) {
				throw new Error("Indexing in progress. Please try again later.");
			}
			
			const queryEmbedding = await this.getEmbedding(query);
			
			// FIX: Check if embedding was successful
			if (!queryEmbedding || queryEmbedding.length === 0) {
				throw new Error("Failed to generate embedding for query");
			}

			// Use the vectorsCache to compare
			const results: SearchResult[] = Array.from(this.vectorsCache.entries())
				.map(([id, vector]) => {
					const similarity = this.cosineSimilarity(queryEmbedding, vector);
					const metadata = this.vectorIndex[id];
					return { 
						id,
						similarity,
						path: metadata.path,
						title: metadata.title,
						content: metadata.content
					};
				})
				.sort((a, b) => b.similarity - a.similarity)
				.slice(0, this.settings.maxResults);

			return results;
		} catch (error) {
			this.logError("Search error", error.message || error.toString());
			console.error("Search error:", error);
			return [];
		}
	}

	// -------------------------------------------------------------------
	// Settings load & save
	// -------------------------------------------------------------------
	async loadSettings() {
		this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
	}

	async saveSettings() {
		await this.saveData(this.settings);
	}

	logError(message: string, details?: string): void {
		// Create a new error log entry
		const errorEntry: ErrorLogEntry = {
			timestamp: Date.now(),
			message: message,
			details: details
		};
		
		// Add to the beginning of the array (newest first)
		this.settings.errorLogs.unshift(errorEntry);
		
		// Keep only the last 100 errors to prevent the log from growing too large
		if (this.settings.errorLogs.length > 100) {
			this.settings.errorLogs = this.settings.errorLogs.slice(0, 100);
		}
		
		// Save settings to persist the error logs
		this.saveSettings();
		
		// Also log to console for immediate debugging
		console.error(`Ollama API Error: ${message}`, details);
	}

	// New method for batch embedding
	async getBatchEmbeddings(texts: string[]): Promise<number[][]> {
		try {
			const response = await fetch(`${this.settings.ollamaEndpoint}/api/embed`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: this.settings.modelName,
					input: texts
				})
			});
			
			if (!response.ok) {
				const errorText = await response.text();
				throw new Error(`Ollama responded with ${response.status} ${response.statusText}: ${errorText}`);
			}
			
			const data = await response.json();
			return data.embeddings || [];
		} catch (err) {
			this.logError(`Failed to get batch embeddings from Ollama`, err.message || err.toString());
			console.error(err);
			// Return an empty array to avoid further errors
			return [];
		}
	}
}

// -------------------------------------------------------------------
// Modal for Searching
// -------------------------------------------------------------------
class SearchModal extends Modal {
	plugin: OllamaSearchPlugin;
	results: SearchResult[] = [];
	searchInput: HTMLInputElement;
	resultsContainer: HTMLDivElement;
	searchTimeout: NodeJS.Timeout | null = null;
	isSearching: boolean = false;
	
	constructor(app: App, plugin: OllamaSearchPlugin) {
		super(app);
		this.plugin = plugin;
	}

	onOpen() {
		const {contentEl} = this;
		contentEl.createEl('h2', {text: 'Ollama Vector Search'});

		// Create search input
		const searchContainer = contentEl.createDiv();
		this.searchInput = searchContainer.createEl('input', {
			type: 'text',
			placeholder: 'Search your notes...'
		});
		this.searchInput.style.width = '100%';
		this.searchInput.style.marginBottom = '10px';

		// Create results container
		this.resultsContainer = contentEl.createDiv();
		this.resultsContainer.style.maxHeight = '400px';
		this.resultsContainer.style.overflow = 'auto';

		// Add initial message
		this.resultsContainer.createEl('div', {
			text: 'Type at least 3 characters to search...',
			cls: 'search-initial-message'
		});

		// Handle search with debouncing
		this.searchInput.addEventListener('input', () => {
			const query = this.searchInput.value;
			
			// Clear any pending search
			if (this.searchTimeout) {
				clearTimeout(this.searchTimeout);
			}
			
			// If query is too short, show message and don't search
			if (query.length < 3) {
				this.clearResults();
				this.resultsContainer.createEl('div', {
					text: 'Type at least 3 characters to search...',
					cls: 'search-initial-message'
				});
				return;
			}
			
			// Show loading indicator
			if (!this.isSearching) {
				this.showLoadingIndicator();
			}
			
			// Debounce the search (wait 300ms after typing stops)
			this.searchTimeout = setTimeout(() => {
				this.performSearch(query);
			}, 300);
		});

		// Focus the input
		this.searchInput.focus();
	}
	
	clearResults() {
		this.resultsContainer.empty();
	}
	
	showLoadingIndicator() {
		this.clearResults();
		const loadingEl = this.resultsContainer.createEl('div', {
			cls: 'search-loading'
		});
		loadingEl.innerHTML = 'Searching<span class="dot-one">.</span><span class="dot-two">.</span><span class="dot-three">.</span>';
	}
	
	async performSearch(query: string) {
		this.isSearching = true;
		
		try {
			this.results = await this.plugin.searchSimilar(query);
			
			this.clearResults();
			
			if (this.results.length === 0) {
				this.resultsContainer.createEl('div', {
					text: 'No results found',
					cls: 'search-no-results'
				});
				return;
			}

			// Create a document fragment to avoid multiple reflows
			const fragment = document.createDocumentFragment();
			
			for (const result of this.results) {
				const resultEl = document.createElement('div');
				resultEl.className = 'search-result';
				resultEl.style.padding = '8px';
				resultEl.style.marginBottom = '8px';
				resultEl.style.border = '1px solid var(--background-modifier-border)';
				resultEl.style.borderRadius = '4px';

				const titleEl = document.createElement('h3');
				titleEl.textContent = result.title;
				resultEl.appendChild(titleEl);
				
				const snippetEl = document.createElement('div');
				snippetEl.className = 'search-result-snippet';
				// Show the first ~150 characters of the chunk
				const snippetText = result.content
					? result.content.substring(0, 150).replace(/\n/g, ' ') + '...'
					: 'No content available';
				snippetEl.textContent = snippetText;
				resultEl.appendChild(snippetEl);

				resultEl.addEventListener('click', () => {
					this.app.workspace.openLinkText(result.path, '');
					this.close();
				});
				
				fragment.appendChild(resultEl);
			}
			
			this.resultsContainer.appendChild(fragment);
			
		} catch (error) {
			this.clearResults();
			
			// Create a more informative error message
			const errorEl = this.resultsContainer.createEl('div', {
				cls: 'search-error'
			});
			
			// Main error message
			errorEl.createEl('p', {
				text: `Error: ${error.message || 'Failed to search'}`
			});
			
			// Add troubleshooting tips
			const tipsList = errorEl.createEl('ul');
			tipsList.createEl('li', {
				text: 'Make sure Ollama is running on your machine'
			});
			tipsList.createEl('li', {
				text: `Check that the model "${this.plugin.settings.modelName}" is available in Ollama`
			});
			tipsList.createEl('li', {
				text: 'Verify your Ollama endpoint in the plugin settings'
			});
			
		} finally {
			this.isSearching = false;
		}
	}

	onClose() {
		if (this.searchTimeout) {
			clearTimeout(this.searchTimeout);
		}
		const {contentEl} = this;
		contentEl.empty();
	}
}

// -------------------------------------------------------------------
// Settings Tab
// -------------------------------------------------------------------
class OllamaSettingTab extends PluginSettingTab {
	plugin: OllamaSearchPlugin;

	constructor(app: App, plugin: OllamaSearchPlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const {containerEl} = this;
		containerEl.empty();

		containerEl.createEl('h2', {text: 'Ollama Vector Search Settings'});

		new Setting(containerEl)
			.setName('Ollama Endpoint')
			.setDesc('URL of your Ollama instance')
			.addText(text => text
				.setPlaceholder('http://localhost:11434')
				.setValue(this.plugin.settings.ollamaEndpoint)
				.onChange(async (value) => {
					this.plugin.settings.ollamaEndpoint = value;
					await this.plugin.saveSettings();
				}));

		new Setting(containerEl)
			.setName('Model Name')
			.setDesc('Ollama model to use for embeddings')
			.addText(text => text
				.setPlaceholder('nomic-embed-text:latest')
				.setValue(this.plugin.settings.modelName)
				.onChange(async (value) => {
					this.plugin.settings.modelName = value;
					await this.plugin.saveSettings();
				}));
				
		new Setting(containerEl)
			.setName('Max Results')
			.setDesc('Maximum number of search results to display')
			.addText(text => text
				.setPlaceholder('5')
				.setValue(this.plugin.settings.maxResults.toString())
				.onChange(async (value) => {
					// FIX: Validate input
					const numValue = parseInt(value);
					if (isNaN(numValue) || numValue < 1) {
						new Notice("Max results must be a positive number");
						return;
					}
					this.plugin.settings.maxResults = numValue;
					await this.plugin.saveSettings();
				}));

		new Setting(containerEl)
			.setName('Chunk Size (characters)')
			.setDesc('Length of each text chunk for embeddings')
			.addText(text => text
				.setPlaceholder('1000')
				.setValue(this.plugin.settings.chunkSize.toString())
				.onChange(async (value) => {
					// FIX: Validate input
					const numValue = parseInt(value);
					if (isNaN(numValue) || numValue < 100) {
						new Notice("Chunk size must be at least 100 characters");
						return;
					}
					this.plugin.settings.chunkSize = numValue;
					await this.plugin.saveSettings();
				}));

		new Setting(containerEl)
			.setName('Chunk Overlap (characters)')
			.setDesc('Overlap between consecutive chunks')
			.addText(text => text
				.setPlaceholder('200')
				.setValue(this.plugin.settings.chunkOverlap.toString())
				.onChange(async (value) => {
					// FIX: Validate input
					const numValue = parseInt(value);
					if (isNaN(numValue) || numValue < 0) {
						new Notice("Chunk overlap must be a non-negative number");
						return;
					}
					this.plugin.settings.chunkOverlap = numValue;
					await this.plugin.saveSettings();
				}));

		// Index status
		new Setting(containerEl)
			.setName('Index Status')
			.setDesc(this.plugin.isIndexed ? 
				`Vector index exists with ${Object.keys(this.plugin.vectorIndex).length} entries` : 
				'No vector index found')
			.addButton(button => button
				.setButtonText(this.plugin.isIndexed ? 'Rebuild Index' : 'Build Index')
				.setCta()
				.setDisabled(this.plugin.isIndexing) // FIX: Disable during indexing
				.onClick(async () => {
					button.setButtonText('Indexing...');
					button.setDisabled(true);

					try {
						// Pass true to force a rebuild when clicking "Rebuild Index"
						const forceRebuild = this.plugin.isIndexed;
						await this.plugin.buildVectorIndex(forceRebuild);
						button.setButtonText(this.plugin.isIndexed ? 'Rebuild Index' : 'Build Index');
						button.setDisabled(false);
					} catch (error) {
						new Notice(`Indexing failed: ${error.message}`);
						button.setButtonText('Retry Indexing');
						button.setDisabled(false);
					}
				}));

		// Error logs section
		containerEl.createEl('h3', {text: 'Error Logs'});
		
		const errorLogContainer = containerEl.createDiv({cls: 'error-log-container'});
		
		if (this.plugin.settings.errorLogs.length === 0) {
			errorLogContainer.createEl('div', {
				text: 'No errors logged',
				cls: 'error-log-empty'
			});
		} else {
			const table = errorLogContainer.createEl('table');
			const headerRow = table.createEl('tr');
			headerRow.createEl('th', {text: 'Time'});
			headerRow.createEl('th', {text: 'Error'});
			headerRow.createEl('th', {text: 'Actions'});
			
			for (const error of this.plugin.settings.errorLogs) {
				const row = table.createEl('tr');
				
				// Format timestamp
				const date = new Date(error.timestamp);
				row.createEl('td', {
					text: date.toLocaleString()
				});
				
				// Error message
				row.createEl('td', {
					text: error.message
				});
				
				// Actions cell
				const actionsCell = row.createEl('td');
				
				// View details button (only if details exist)
				if (error.details) {
					const viewButton = actionsCell.createEl('button', {
						text: 'Details',
						cls: 'mod-cta'
					});
					viewButton.style.marginRight = '5px';
					viewButton.addEventListener('click', () => {
						// Toggle details row
						const detailsId = `error-${error.timestamp}`;
						const existingDetails = table.querySelector(`#${detailsId}`);
						
						if (existingDetails) {
							existingDetails.remove();
						} else {
							const detailsRow = table.createEl('tr', {
								cls: 'error-details-row'
							});
							detailsRow.id = detailsId;
							
							const detailsCell = detailsRow.createEl('td', {
								attr: {colspan: '3'}
							});
							
							const pre = detailsCell.createEl('pre');
							pre.createEl('code', {
								text: error.details
							});
							
							// Insert after the current row
							row.after(detailsRow);
						}
					});
				}
			}
			
			// Add clear logs button
			const clearButtonContainer = errorLogContainer.createDiv();
			clearButtonContainer.style.marginTop = '10px';
			clearButtonContainer.style.textAlign = 'right';
			
			const clearButton = clearButtonContainer.createEl('button', {
				text: 'Clear Error Logs',
				cls: 'mod-warning'
			});
			
			clearButton.addEventListener('click', async () => {
				this.plugin.settings.errorLogs = [];
				await this.plugin.saveSettings();
				this.display(); // Refresh the settings tab
			});
		}
	}
}







 