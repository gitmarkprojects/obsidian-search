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
				new SearchModal(this.app, this).open();
			}
		});

		// Add settings tab
		this.addSettingTab(new OllamaSettingTab(this.app, this));

		// Load index and vectors if they exist
		if (this.isIndexed) {
			await this.loadVectorIndex();
			this.vectorsCache = await this.loadVectorsFromBinary(); // Cache it now
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
		
		const files = this.app.vault.getMarkdownFiles();
		let indexingNotice = new Notice(`Indexing 0 / ${files.length} notes...`, 0);

		// If forcing a full rebuild, clear the existing index first
		if (this.forceReindex) {
			this.vectorIndex = {};
			this.vectorsCache.clear();
		}

		for (const [i, file] of files.entries()) {
			const stat = await this.app.vault.adapter.stat(file.path);
			if (!stat) {
				// Can't retrieve file stats; skip
				continue;
			}

			// Check if this file is already indexed and up to date
			const oldChunks = Object.entries(this.vectorIndex)
				.filter(([id, entry]) => entry.path === file.path);

			// Only skip if not forcing reindex AND the file is up to date
			const isUpToDate = !this.forceReindex && 
				oldChunks.length > 0 && 
				oldChunks.every(([id, entry]) => entry.mtime === stat.mtime);
				
			if (isUpToDate) {
				continue;
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

					// Generate a unique ID for each chunk
					const id = this.generateId();

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

		indexingNotice.hide();
		new Notice('Indexing complete!');

		// Save vectors to binary file
		await this.saveVectorsToBinary(
			Array.from(this.vectorsCache.entries()).map(([id, vector]) => ({ id, vector }))
		);

		// Save index
		await this.saveVectorIndex();
		this.isIndexed = true;
	}

	// -------------------------------------------------------------------
	// Text chunking logic
	// -------------------------------------------------------------------
	private chunkText(text: string, size: number, overlap: number): string[] {
		// Splits text into overlapping chunks of length `size` with `overlap` characters
		// of overlap (except possibly the last chunk if the text ends).
		const chunks: string[] = [];
		let start = 0;

		while (start < text.length) {
			const end = start + size;
			const chunk = text.slice(start, end);
			chunks.push(chunk);

			// Move start forward but incorporate overlap
			start += (size - overlap);
		}
		return chunks;
	}

	// -------------------------------------------------------------------
	// Random ID generator (for chunk entries)
	// -------------------------------------------------------------------
	generateId(): string {
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
			while (offset < arrayBuffer.byteLength) {
				// Read id length
				const idLength = view.getUint16(offset, true);
				offset += 2;

				// Read id
				let id = '';
				for (let i = 0; i < idLength; i++) {
					id += String.fromCharCode(view.getUint8(offset + i));
				}
				offset += idLength;

				// Align offset back to multiple of 4
				while (offset % 4 !== 0) {
					offset++;
				}

				// Read vector length
				const vectorLength = view.getUint32(offset, true);
				offset += 4;

				// Read float32 array
				const vector = new Array(vectorLength);
				for (let i = 0; i < vectorLength; i++) {
					vector[i] = view.getFloat32(offset, true);
					offset += 4;
				}

				vectors.set(id, vector);
			}

			console.log(`Loaded ${vectors.size} vectors from binary file`);
			return vectors;
		} catch (error) {
			console.error("Error loading vectors from binary:", error);
			throw error;
		}
	}

	// -------------------------------------------------------------------
	// Embedding & Similarity
	// -------------------------------------------------------------------
	async getEmbedding(text: string): Promise<number[]> {
		try {
			const response = await fetch(`${this.settings.ollamaEndpoint}/api/embeddings`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: this.settings.modelName,
					prompt: text
				})
			});
			if (!response.ok) {
				throw new Error(`Ollama responded with ${response.status} ${response.statusText}`);
			}
			const data = await response.json();
			return data.embedding;
		} catch (err) {
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
		return dotProduct / (magA * magB);
	}

	// -------------------------------------------------------------------
	// Main search routine
	// -------------------------------------------------------------------
	async searchSimilar(query: string): Promise<SearchResult[]> {
		try {
			const queryEmbedding = await this.getEmbedding(query);

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
			this.resultsContainer.createEl('div', {
				text: `Error: ${error.message || 'Failed to search'}`,
				cls: 'search-error'
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
					this.plugin.settings.maxResults = parseInt(value) || 5;
					await this.plugin.saveSettings();
				}));

		new Setting(containerEl)
			.setName('Chunk Size (characters)')
			.setDesc('Length of each text chunk for embeddings')
			.addText(text => text
				.setPlaceholder('1000')
				.setValue(this.plugin.settings.chunkSize.toString())
				.onChange(async (value) => {
					this.plugin.settings.chunkSize = parseInt(value) || 1000;
					await this.plugin.saveSettings();
				}));

		new Setting(containerEl)
			.setName('Chunk Overlap (characters)')
			.setDesc('Overlap between consecutive chunks')
			.addText(text => text
				.setPlaceholder('200')
				.setValue(this.plugin.settings.chunkOverlap.toString())
				.onChange(async (value) => {
					this.plugin.settings.chunkOverlap = parseInt(value) || 200;
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
				.onClick(async () => {
					button.setButtonText('Indexing...');
					button.setDisabled(true);

					try {
						// Pass true to force a rebuild when clicking "Rebuild Index"
						const forceRebuild = this.plugin.isIndexed;
						await this.plugin.buildVectorIndex(forceRebuild);
						button.setButtonText(this.plugin.isIndexed ? 'Rebuild Index' : 'Build Index');
						button.setDisabled(false);
						this.display(); // Refresh
					} catch (error) {
						new Notice('Error building index: ' + error);
						button.setButtonText('Retry');
						button.setDisabled(false);
					}
				}));

		// Error logs section
		containerEl.createEl('h3', {text: 'Error Logs'});

		const errorLogContainer = containerEl.createDiv('error-log-container');
		errorLogContainer.style.maxHeight = '300px';
		errorLogContainer.style.overflow = 'auto';
		errorLogContainer.style.border = '1px solid var(--background-modifier-border)';
		errorLogContainer.style.borderRadius = '4px';
		errorLogContainer.style.padding = '8px';
		errorLogContainer.style.marginBottom = '16px';

		if (this.plugin.settings.errorLogs.length === 0) {
			errorLogContainer.createEl('p', {
				text: 'No errors logged yet.',
				cls: 'error-log-empty'
			});
		} else {
			// Create a table for the logs
			const table = errorLogContainer.createEl('table');
			table.style.width = '100%';
			table.style.borderCollapse = 'collapse';
			
			// Table header
			const thead = table.createEl('thead');
			const headerRow = thead.createEl('tr');
			headerRow.createEl('th', {text: 'Time'}).style.textAlign = 'left';
			headerRow.createEl('th', {text: 'Error'}).style.textAlign = 'left';
			
			// Table body
			const tbody = table.createEl('tbody');
			
			for (const log of this.plugin.settings.errorLogs) {
				const row = tbody.createEl('tr');
				row.style.borderBottom = '1px solid var(--background-modifier-border)';
				
				// Format timestamp
				const date = new Date(log.timestamp);
				const timeCell = row.createEl('td');
				timeCell.style.padding = '4px';
				timeCell.style.whiteSpace = 'nowrap';
				timeCell.textContent = date.toLocaleString();
				
				// Error message
				const messageCell = row.createEl('td');
				messageCell.style.padding = '4px';
				messageCell.textContent = log.message;
				
				// Make the row expandable if there are details
				if (log.details) {
					row.style.cursor = 'pointer';
					row.addEventListener('click', () => {
						// Check if details row already exists
						const nextRow = row.nextElementSibling;
						if (nextRow && nextRow.classList.contains('error-details-row')) {
							nextRow.remove();
						} else {
							const detailsRow = tbody.createEl('tr');
							detailsRow.classList.add('error-details-row');
							detailsRow.style.backgroundColor = 'var(--background-secondary)';
							
							const detailsCell = detailsRow.createEl('td', {
								attr: { colspan: '2' }
							});
							detailsCell.style.padding = '8px';
							detailsCell.style.fontFamily = 'monospace';
							detailsCell.style.whiteSpace = 'pre-wrap';
							detailsCell.textContent = log.details ?? '';
							
							row.after(detailsRow);
						}
					});
				}
			}
		}

		// Add button to clear logs
		new Setting(containerEl)
			.setName('Clear Error Logs')
			.setDesc('Remove all error logs')
			.addButton(button => button
				.setButtonText('Clear Logs')
				.onClick(async () => {
					this.plugin.settings.errorLogs = [];
					await this.plugin.saveSettings();
					this.display(); // Refresh the settings view
				}));
	}
}
