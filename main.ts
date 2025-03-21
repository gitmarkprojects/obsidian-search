import { App, MarkdownView, Modal, Notice, Plugin, PluginSettingTab, Setting, TFile } from 'obsidian';
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
	autoIndex: boolean;        // New: auto-index on file changes
	excludeFolders: string[]; // New: folders to exclude from indexing
}

interface VectorIndexEntry {
	path: string;
	title: string;
	mtime: number;         // Last modified time
	chunkIndex: number;    // Which chunk number of the file
	content: string;       // The chunk text
	blockRef?: string;     // New: block reference ID for navigation
	tags?: string[];       // New: tags associated with this chunk's note
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
	blockRef?: string;
}

interface ErrorLogEntry {
	timestamp: number;
	message: string;
	details?: string;
}

// Info per file for indexing
interface FileInfo {
	mtime: number;
	tags: Set<string>;
	ids: string[];
}

const DEFAULT_SETTINGS: OllamaSearchPluginSettings = {
	ollamaEndpoint: 'http://localhost:11434',
	modelName: 'jeffh/intfloat-multilingual-e5-large-instruct:q8_0',
	maxResults: 5,
	chunkSize: 1200,      // Default chunk size in characters
	chunkOverlap: 200,    // Default overlap in characters
	errorLogs: [],        // Initialize empty error logs
	autoIndex: false,
	excludeFolders: []
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
	private isSearchQuery: boolean = false;
	private fileInfoMap: Map<string, FileInfo> = new Map();

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
		if (this.isIndexed) {
			this.initializeFileInfoMap();
		}
		// Register vault events for incremental indexing
		this.registerEvent(this.app.vault.on('modify', (file: TFile) => {
			if (!this.settings.autoIndex || file.extension !== 'md') return;
			this.indexFile(file);
		}));
		this.registerEvent(this.app.vault.on('create', (file: TFile) => {
			if (!this.settings.autoIndex || file.extension !== 'md') return;
			this.indexFile(file);
		}));
		this.registerEvent(this.app.vault.on('delete', (file: TFile) => {
			if (!this.settings.autoIndex || file.extension !== 'md') return;
			this.removeFromIndex(file.path);
		}));
		this.registerEvent(this.app.vault.on('rename', (file: TFile, oldPath: string) => {
			if (file.extension !== 'md') return;
			this.updateIndexForRename(oldPath, file.path, file);
		}));
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
			this.fileInfoMap.clear();
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

	// Initialize or rebuild file information map from vectorIndex
	private initializeFileInfoMap(): void {
		this.fileInfoMap.clear();
		for (const [id, entry] of Object.entries(this.vectorIndex)) {
			const filePath = entry.path;
			let info = this.fileInfoMap.get(filePath);
			if (!info) {
				info = {
					mtime: entry.mtime,
					tags: new Set(entry.tags ?? []),
					ids: []
				};
				this.fileInfoMap.set(filePath, info);
			}
			info.ids.push(id);
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
			this.fileInfoMap.clear();
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

					// NEW: get all chunk embeddings in one batch
					const embeddings = await this.getBatchEmbeddings(chunks);

					// Validate length to avoid mismatch
					if (embeddings.length !== chunks.length) {
						console.warn(
							`Batch embeddings returned ${embeddings.length} vectors for ${chunks.length} chunks in file: ${file.path}`
						);
					}

					// Store each embedding
					const fileIds: string[] = [];
					for (let cIndex = 0; cIndex < chunks.length; cIndex++) {
						const chunk = chunks[cIndex];
						const embedding = embeddings[cIndex] ?? [];

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

						// Generate a block reference ID for this chunk
						const blockRef = `^chunk-${this.generateId().substring(0, 8)}`;

						// Store in memory
						this.vectorsCache.set(id, embedding);

						// Update index
						this.vectorIndex[id] = {
							path: file.path,
							title: file.basename,
							mtime: stat.mtime,
							chunkIndex: cIndex,
							content: chunk,
							blockRef: blockRef,
							tags: []
						};
						fileIds.push(id);
					}
					// Update file info map for this file
					const tags = this.extractTagsFromContent(content);
					this.fileInfoMap.set(file.path, { mtime: stat.mtime, tags: new Set(tags), ids: fileIds });

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
				// Only add chunk if it passes the entropy filter
				if (this.hasAdequateEntropy(currentChunk)) {
					chunks.push(currentChunk);
				}
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
				// Only add chunk if it passes the entropy filter
				if (this.hasAdequateEntropy(chunkToAdd)) {
					chunks.push(chunkToAdd);
				}
				currentChunk = currentChunk.slice(size - overlap);
			}
		}
		
		// Add the last chunk if it's not empty and passes the filter
		if (currentChunk.length > 0 && this.hasAdequateEntropy(currentChunk)) {
			chunks.push(currentChunk);
		}
		
		return chunks;
	}

	// NEW: Calculate Shannon entropy of text to detect repetitive content
	private hasAdequateEntropy(text: string): boolean {
		// Minimum text length to consider
		if (text.length < 20) return true;
		
		// Calculate character-level entropy
		const charEntropy = this.calculateEntropy(text);
		
		// Calculate word-level entropy for additional insight
		const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
		const wordEntropy = words.length > 5 ? this.calculateEntropy(words) : 0;
		
		// Thresholds based on empirical testing
		// Character entropy below 3.0 bits usually indicates highly repetitive text
		// For reference: random English text typically has 4.0-4.5 bits of entropy per character
		const charEntropyThreshold = 3.0;
		
		// Word entropy below 2.5 bits usually indicates repetitive word usage
		// For reference: normal English text typically has 7-10 bits of entropy per word
		// We're using a lower threshold since we're calculating on a small sample
		const wordEntropyThreshold = 2.5;
		
		// For very short texts, we mainly rely on character entropy
		if (words.length < 10) {
			return charEntropy >= charEntropyThreshold;
		}
		
		// For longer texts, use both character and word entropy
		return charEntropy >= charEntropyThreshold || wordEntropy >= wordEntropyThreshold;
	}

	// Calculate Shannon entropy of a string or array of strings
	private calculateEntropy(input: string | string[]): number {
		// Create frequency map
		const frequencyMap = new Map<string, number>();
		let totalElements = 0;
		
		if (typeof input === 'string') {
			// Character-level entropy
			for (const char of input) {
				if (char.trim() === '') continue; // Skip whitespace for character entropy
				totalElements++;
				frequencyMap.set(char, (frequencyMap.get(char) || 0) + 1);
			}
		} else {
			// Word-level entropy
			for (const word of input) {
				totalElements++;
				frequencyMap.set(word, (frequencyMap.get(word) || 0) + 1);
			}
		}
		
		if (totalElements === 0) return 0;
		
		// Calculate Shannon entropy: -sum(p(x) * log2(p(x)))
		let entropy = 0;
		for (const count of frequencyMap.values()) {
			const probability = count / totalElements;
			entropy -= probability * Math.log2(probability);
		}
		
		return entropy;
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
	private normalizeText(text: string): string {
		// Step 1: Convert to lowercase
		let normalized = text.toLowerCase();
		
		// Step 2: Remove special characters, keeping alphanumeric, spaces, and some punctuation
		// This regex keeps letters, numbers, spaces, periods, commas, and basic punctuation
		// but removes unusual symbols, emojis, etc.
		normalized = normalized.replace(/[^\w\s.,?!;:()\-'"]/g, ' ');
		
		// Step 3: Replace multiple spaces with a single space
		normalized = normalized.replace(/\s+/g, ' ');
		
		return normalized.trim();
	}

	async getEmbedding(text: string, retryCount = 0): Promise<number[]> {
		const maxRetries = 2; // Allow up to 2 retries
		
		try {
			// Normalize the text before getting embedding
			const normalizedText = this.normalizeText(text);
			
			// Apply instruction template for search queries
			let inputText = normalizedText;
			if (this.isSearchQuery) {
				inputText = `Instruction: Retrieve a similar sentence. Query: ${normalizedText}`;
			}
			
			const response = await fetch(`${this.settings.ollamaEndpoint}/api/embed`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: this.settings.modelName,
					input: inputText
				})
			});
			if (!response.ok) {
				const errorText = await response.text();
				throw new Error(`Ollama responded with ${response.status} ${response.statusText}: ${errorText}`);
			}
			const data = await response.json();
			
			// The response format is arrays of embeddings; we want the first
			return data.embeddings?.[0] || [];
		} catch (err) {
			this.logError(`Failed to get embedding from Ollama`, err.message || err.toString());
			new Notice(`Failed to get embedding from Ollama: ${err.message}`);
			console.error(err);
			// Return an empty embedding to avoid further errors
			return [];
		}
	}

	cosineSimilarity(vecA: number[], vecB: number[]): number {
		// Basic guard for length mismatch
		if (vecA.length !== vecB.length || vecA.length === 0) {
			return 0;
		}
		
		// Calculate dot product while handling NaN values
		let dotProduct = 0;
		let magASquared = 0;
		let magBSquared = 0;
		
		for (let i = 0; i < vecA.length; i++) {
			// Skip NaN values
			if (isNaN(vecA[i]) || isNaN(vecB[i])) {
				continue;
			}
			
			dotProduct += vecA[i] * vecB[i];
			magASquared += vecA[i] * vecA[i];
			magBSquared += vecB[i] * vecB[i];
		}
		
		// Calculate magnitudes
		const magA = Math.sqrt(magASquared);
		const magB = Math.sqrt(magBSquared);
		
		// Handle zero or very small magnitudes with a small epsilon value
		const epsilon = 1e-10;
		if (magA < epsilon || magB < epsilon) {
			return 0;
		}
		
		// Calculate and bound similarity to [-1, 1] range to handle floating point errors
		const similarity = dotProduct / (magA * magB);
		return Math.max(-1, Math.min(1, similarity));
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
			
			// Set the flag to indicate we're processing a search query
			this.isSearchQuery = true;
			
			const queryEmbedding = await this.getEmbedding(query);
			
			// Reset the flag
			this.isSearchQuery = false;
			
			// FIX: Check if embedding was successful
			if (!queryEmbedding || queryEmbedding.length === 0) {
				throw new Error("Failed to generate embedding for query");
			}

			// Minimum length check
			if (query.trim().length < 3) {
				throw new Error('Search query must be at least 3 characters long');
			}

			// Use the vectorsCache to compare
			const results: SearchResult[] = [];
			for (const [id, vector] of this.vectorsCache.entries()) {
				const meta = this.vectorIndex[id];
				const similarity = this.cosineSimilarity(queryEmbedding, vector);
				results.push({ id, similarity, path: meta.path, title: meta.title, content: meta.content, blockRef: meta.blockRef });
			}
			results.sort((a, b) => b.similarity - a.similarity);
			if (results.length > this.settings.maxResults) {
				results.length = this.settings.maxResults;
			}
			return results;
		} catch (error) {
			// Make sure to reset the flag even if there's an error
			this.isSearchQuery = false;
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
			// For document chunks, we don't need to add instructions
			// The isSearchQuery flag should be false here
			this.isSearchQuery = false;
			
			// Normalize each text in the batch
			const normalizedTexts = texts.map(text => this.normalizeText(text));
			
			const response = await fetch(`${this.settings.ollamaEndpoint}/api/embed`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					model: this.settings.modelName,
					input: normalizedTexts
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

	// Incremental indexing helper methods
	private async indexFile(file: TFile): Promise<void> {
		try {
			const stat = await this.app.vault.adapter.stat(file.path);
			if (!stat) return;
			// Check if file already indexed and up to date
			const oldIds = (this.fileInfoMap.get(file.path)?.ids) || [];
			if (oldIds.length > 0 && this.fileInfoMap.get(file.path)?.mtime === stat.mtime) {
				return; // no update needed
			}
			// Remove old index entries for this file
			for (const id of oldIds) {
				this.vectorsCache.delete(id);
				delete this.vectorIndex[id];
			}
			this.fileInfoMap.delete(file.path);
			const content = await this.app.vault.read(file);
			const chunks = this.chunkText(content, this.settings.chunkSize, this.settings.chunkOverlap);
			const embeddings = await this.getBatchEmbeddings(chunks);
			if (embeddings.length !== chunks.length) {
				console.warn(`Embedding count mismatch for ${file.path}: got ${embeddings.length}, expected ${chunks.length}`);
			}
			const newIds: string[] = [];
			for (let cIndex = 0; cIndex < chunks.length; cIndex++) {
				const embedding = embeddings[cIndex] ?? [];
				if (embedding.length === 0) {
					continue;
				}
				let id: string;
				do {
					id = this.generateId();
				} while (this.vectorIndex[id]);
				const blockRef = `^chunk-${this.generateId().substring(0, 8)}`;
				this.vectorsCache.set(id, embedding);
				this.vectorIndex[id] = {
					path: file.path,
					title: file.basename,
					mtime: stat.mtime,
					chunkIndex: cIndex,
					content: chunks[cIndex],
					blockRef: blockRef,
					tags: []
				};
				newIds.push(id);
			}
			// Update fileInfoMap
			const tags = this.extractTagsFromContent(content);
			this.fileInfoMap.set(file.path, { mtime: stat.mtime, tags: new Set(tags), ids: newIds });
			// Persist updated index
			await this.saveVectorIndex();
			await this.saveVectorsToBinary(Array.from(this.vectorsCache.entries()).map(([id, vector]) => ({id, vector})));
		} catch (error) {
			this.logError(`Error indexing file ${file.path}`, error.message);
			console.error(`Error indexing file ${file.path}:`, error);
		}
	}

	private async removeFromIndex(filePath: string): Promise<void> {
		const info = this.fileInfoMap.get(filePath);
		if (!info) return;
		for (const id of info.ids) {
			this.vectorsCache.delete(id);
			delete this.vectorIndex[id];
		}
		this.fileInfoMap.delete(filePath);
		try {
			await this.saveVectorIndex();
			await this.saveVectorsToBinary(Array.from(this.vectorsCache.entries()).map(([id, vector]) => ({id, vector})));
		} catch (error) {
			this.logError(`Error removing index for file ${filePath}`, error.message);
		}
	}

	private updateIndexForRename(oldPath: string, newPath: string, file: TFile): void {
		const info = this.fileInfoMap.get(oldPath);
		if (!info) return;
		// Update vectorIndex entries
		for (const id of info.ids) {
			const entry = this.vectorIndex[id];
			if (entry) {
				entry.path = newPath;
				entry.title = file.basename;
			}
		}
		// Update fileInfoMap key
		this.fileInfoMap.delete(oldPath);
		info.mtime = file.stat.mtime;
		this.fileInfoMap.set(newPath, info);
		try {
			this.saveVectorIndex();
		} catch (error) {
			this.logError(`Error updating index for rename ${oldPath} -> ${newPath}`, error.message);
		}
	}

	// Extract tags from note content (including YAML frontmatter)
	private extractTagsFromContent(text: string): string[] {
		const tags: string[] = [];
		// Extract tags from YAML frontmatter
		if (text.startsWith('---')) {
			const end = text.indexOf('\n---', 3);
			if (end !== -1) {
				const frontmatter = text.substring(0, end + 4);
				const tagLines = frontmatter.match(/^tags:\s*(.+)$/m);
				if (tagLines) {
					const tagsLine = tagLines[1];
					// Remove brackets and quotes, then split by commas or spaces
					const clean = tagsLine.replace(/\[|\]|"|'/g, '');
					tags.push(...clean.split(/[\s,]+/).filter(t => t));
				}
			}
		}
		// Extract inline #tags from content
		const inlineTags = text.match(/#([A-Za-z0-9_\-\/]+)/g);
		if (inlineTags) {
			for (const tag of inlineTags) {
				tags.push(tag.startsWith('#') ? tag.substring(1) : tag);
			}
		}
		return Array.from(new Set(tags.map(t => t.toLowerCase())));
	}
}

// -------------------------------------------------------------------
// Search Modal
// -------------------------------------------------------------------
class SearchModal extends Modal {
    plugin: OllamaSearchPlugin;
    query: string = '';
    results: SearchResult[] = [];
    isLoading: boolean = false;
    errorMessage: string = '';

    constructor(app: App, plugin: OllamaSearchPlugin) {
        super(app);
        this.plugin = plugin;
    }

    onOpen() {
        const { contentEl } = this;
        contentEl.empty();
        contentEl.addClass('ollama-search-modal');

        // Create search input
        const searchContainer = contentEl.createDiv({ cls: 'search-input-container' });
        const searchInput = searchContainer.createEl('input', {
            type: 'text',
            placeholder: 'Search your notes semantically...'
        });
        searchInput.focus();

        // Create results container
        const resultsContainer = contentEl.createDiv({ cls: 'search-results-container' });
        resultsContainer.createDiv({ 
            cls: 'search-initial-message',
            text: 'Enter a search query to find semantically similar content in your notes.'
        });

        // Handle search input
        searchInput.addEventListener('input', async () => {
            this.query = searchInput.value;
            if (this.query.length < 3) {
                resultsContainer.empty();
                resultsContainer.createDiv({ 
                    cls: 'search-initial-message',
                    text: 'Enter at least 3 characters to search.'
                });
                return;
            }

            // Show loading indicator
            resultsContainer.empty();
            const loadingDiv = resultsContainer.createDiv({ cls: 'search-loading' });
            loadingDiv.setText('Searching');
            loadingDiv.createSpan({ cls: 'dot-one', text: '.' });
            loadingDiv.createSpan({ cls: 'dot-two', text: '.' });
            loadingDiv.createSpan({ cls: 'dot-three', text: '.' });
            
            this.isLoading = true;
            this.errorMessage = '';

            try {
                // Debounce search to avoid too many requests
                setTimeout(async () => {
                    if (searchInput.value === this.query) {
                        this.results = await this.plugin.searchSimilar(this.query);
                        this.renderResults(resultsContainer);
                    }
                }, 300);
            } catch (error) {
                this.errorMessage = error.message || 'An error occurred during search';
                this.renderResults(resultsContainer);
            }
        });
    }

    renderResults(container: HTMLElement) {
        container.empty();
        this.isLoading = false;

        // Handle error state
        if (this.errorMessage) {
            container.createDiv({ 
                cls: 'search-error', 
                text: `Error: ${this.errorMessage}` 
            });
            return;
        }

        // Handle empty results
        if (this.results.length === 0) {
            container.createDiv({ 
                cls: 'search-no-results', 
                text: 'No results found.' 
            });
            return;
        }

        // Show result count
        container.createDiv({ 
            cls: 'search-results-count', 
            text: `${this.results.length} result${this.results.length !== 1 ? 's' : ''}`
        });
        
        // Create results list
        const resultsList = container.createDiv({ cls: 'search-results-list' });
        
        // Render results
        for (const result of this.results) {
            const resultCard = resultsList.createDiv({ cls: 'search-result-card' });
            
            // Title with simple score
            const titleDiv = resultCard.createDiv({ cls: 'search-result-title' });
            titleDiv.createSpan({ text: result.title });
            
            // Add simple score
            const scoreValue = (result.similarity * 100).toFixed(0);
            titleDiv.createSpan({ 
                cls: 'search-result-score',
                text: `${scoreValue}%`
            });
            
            // Path
            resultCard.createDiv({ 
                cls: 'search-result-path', 
                text: result.path 
            });
            
            // Content preview
            const contentDiv = resultCard.createDiv({ cls: 'search-result-content' });
            
            // Format and truncate content
            let snippetText = result.content.trim();
            if (snippetText.length > 300) {
                const breakPoint = snippetText.lastIndexOf(' ', 300);
                snippetText = snippetText.substring(0, breakPoint > 0 ? breakPoint : 300) + '...';
            }
            contentDiv.setText(snippetText);
            
            // Handle click to open the file
            resultCard.addEventListener('click', () => {
                this.openResult(result);
            });
        }
    }
    
    // Helper method to open a result
    private openResult(result: SearchResult): void {
        // Navigate to the file and position
        const targetFile = this.app.vault.getAbstractFileByPath(result.path);
        if (targetFile instanceof TFile) {
            this.app.workspace.getLeaf().openFile(targetFile).then(() => {
                // If we have a block reference, scroll to it
                if (result.blockRef) {
                    // Use setTimeout to ensure the editor is loaded
                    setTimeout(() => {
                        const currentView = this.app.workspace.getActiveViewOfType(MarkdownView);
                        if (currentView) {
                            const editor = currentView.editor;
                            const content = editor.getValue();
                            const blockPosition = content.indexOf(result.blockRef ?? '');
                            
                            if (blockPosition !== -1) {
                                // Calculate line number
                                const contentToBlock = content.substring(0, blockPosition);
                                const lineCount = contentToBlock.split('\n').length - 1;
                                
                                // Scroll to position
                                editor.setCursor({ line: lineCount, ch: 0 });
                                editor.scrollIntoView({ from: { line: lineCount, ch: 0 }, to: { line: lineCount, ch: 0 } }, true);
                            }
                        }
                    }, 100);
                }
            });
            this.close();
        }
    }

    onClose() {
        const { contentEl } = this;
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
        const { containerEl } = this;
        containerEl.empty();

        containerEl.createEl('h2', { text: 'Ollama Vector Search Settings' });

        // Ollama Endpoint
        new Setting(containerEl)
            .setName('Ollama Endpoint')
            .setDesc('The URL of your Ollama server')
            .addText(text => text
                .setPlaceholder('http://localhost:11434')
                .setValue(this.plugin.settings.ollamaEndpoint)
                .onChange(async (value) => {
                    this.plugin.settings.ollamaEndpoint = value;
                    await this.plugin.saveSettings();
                }));

        // Model Name
        new Setting(containerEl)
            .setName('Embedding Model')
            .setDesc('The name of the embedding model to use')
            .addText(text => text
                .setPlaceholder('jeffh/intfloat-multilingual-e5-large-instruct:q8_0')
                .setValue(this.plugin.settings.modelName)
                .onChange(async (value) => {
                    this.plugin.settings.modelName = value;
                    await this.plugin.saveSettings();
                }));

        // Max Results
        new Setting(containerEl)
            .setName('Max Results')
            .setDesc('Maximum number of search results to display')
            .addSlider(slider => slider
                .setLimits(1, 20, 1)
                .setValue(this.plugin.settings.maxResults)
                .setDynamicTooltip()
                .onChange(async (value) => {
                    this.plugin.settings.maxResults = value;
                    await this.plugin.saveSettings();
                }));

        // Chunk Size
        new Setting(containerEl)
            .setName('Chunk Size')
            .setDesc('Size of text chunks in characters')
            .addSlider(slider => slider
                .setLimits(500, 3000, 100)
                .setValue(this.plugin.settings.chunkSize)
                .setDynamicTooltip()
                .onChange(async (value) => {
                    this.plugin.settings.chunkSize = value;
                    await this.plugin.saveSettings();
                }));

        // Chunk Overlap
        new Setting(containerEl)
            .setName('Chunk Overlap')
            .setDesc('Overlap between chunks in characters')
            .addSlider(slider => slider
                .setLimits(0, 500, 50)
                .setValue(this.plugin.settings.chunkOverlap)
                .setDynamicTooltip()
                .onChange(async (value) => {
                    this.plugin.settings.chunkOverlap = value;
                    await this.plugin.saveSettings();
                }));

        // Auto-index toggle
        new Setting(containerEl)
            .setName('Auto-Index')
            .setDesc('Automatically update index when files change')
            .addToggle(toggle => toggle
                .setValue(this.plugin.settings.autoIndex)
                .onChange(async (value) => {
                    this.plugin.settings.autoIndex = value;
                    await this.plugin.saveSettings();
                }));

        // Excluded Folders
        new Setting(containerEl)
            .setName('Excluded Folders')
            .setDesc('Folders to exclude from indexing (comma-separated)')
            .addText(text => text
                .setPlaceholder('templates, attachments')
                .setValue(this.plugin.settings.excludeFolders.join(', '))
                .onChange(async (value) => {
                    this.plugin.settings.excludeFolders = value.split(',').map(s => s.trim()).filter(s => s);
                    await this.plugin.saveSettings();
                }));

        // Index Building
        containerEl.createEl('h3', { text: 'Index Management' });

        const indexStatus = containerEl.createDiv({ cls: 'index-status' });
        if (this.plugin.isIndexed) {
            indexStatus.createSpan({ text: 'Index Status: ', cls: 'index-status-label' });
            indexStatus.createSpan({ 
                text: `Indexed (${Object.keys(this.plugin.vectorIndex).length} chunks)`, 
                cls: 'index-status-value' 
            });
        } else {
            indexStatus.createSpan({ text: 'Index Status: ', cls: 'index-status-label' });
            indexStatus.createSpan({ text: 'Not indexed', cls: 'index-status-value' });
        }

        // Build Index Button
        new Setting(containerEl)
            .setName('Build Index')
            .setDesc('Build or rebuild the vector index')
            .addButton(button => button
                .setButtonText(this.plugin.isIndexed ? 'Rebuild Index' : 'Build Index')
                .setCta()
                .onClick(async () => {
                    if (this.plugin.isIndexing) {
                        new Notice('Indexing already in progress');
                        return;
                    }
                    
                    try {
                        button.setButtonText('Indexing...');
                        button.setDisabled(true);
                        
                        await this.plugin.buildVectorIndex(true);
                        
                        button.setButtonText('Rebuild Index');
                        button.setDisabled(false);
                        
                        // Update index status
                        indexStatus.empty();
                        indexStatus.createSpan({ text: 'Index Status: ', cls: 'index-status-label' });
                        indexStatus.createSpan({ 
                            text: `Indexed (${Object.keys(this.plugin.vectorIndex).length} chunks)`, 
                            cls: 'index-status-value' 
                        });
                    } catch (error) {
                        button.setButtonText(this.plugin.isIndexed ? 'Rebuild Index' : 'Build Index');
                        button.setDisabled(false);
                        new Notice(`Indexing failed: ${error.message}`);
                    }
                }));

        // Error Logs
        containerEl.createEl('h3', { text: 'Error Logs' });
        
        const errorLogContainer = containerEl.createDiv({ cls: 'error-log-container' });
        
        if (this.plugin.settings.errorLogs.length === 0) {
            errorLogContainer.createDiv({ 
                cls: 'error-log-empty',
                text: 'No errors logged'
            });
        } else {
            const table = errorLogContainer.createEl('table');
            const headerRow = table.createEl('tr');
            headerRow.createEl('th', { text: 'Time' });
            headerRow.createEl('th', { text: 'Error' });
            
            // Show the 10 most recent errors
            const recentErrors = this.plugin.settings.errorLogs.slice(0, 10);
            
            for (const error of recentErrors) {
                const row = table.createEl('tr');
                
                // Format timestamp
                const date = new Date(error.timestamp);
                const formattedDate = `${date.toLocaleDateString()} ${date.toLocaleTimeString()}`;
                
                row.createEl('td', { text: formattedDate });
                row.createEl('td', { text: error.message });
                
                // If there are details, add them in a collapsible row
                if (error.details) {
                    const detailsRow = table.createEl('tr', { cls: 'error-details-row' });
                    const detailsCell = detailsRow.createEl('td', { attr: { colspan: '2' } });
                    detailsCell.createEl('pre', { text: error.details });
                    
                    // Initially hide details
                    detailsRow.style.display = 'none';
                    
                    // Toggle details on click
                    row.addEventListener('click', () => {
                        detailsRow.style.display = detailsRow.style.display === 'none' ? 'table-row' : 'none';
                    });
                    
                    // Add pointer cursor to indicate clickable
                    row.style.cursor = 'pointer';
                }
            }
            
            // Add clear logs button
            new Setting(containerEl)
                .setName('Clear Error Logs')
                .setDesc('Remove all error logs')
                .addButton(button => button
                    .setButtonText('Clear Logs')
                    .onClick(async () => {
                        this.plugin.settings.errorLogs = [];
                        await this.plugin.saveSettings();
                        this.display(); // Refresh the display
                    }));
        }
    }
}
