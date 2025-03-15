# Ollama Vector Search for Obsidian

This plugin enables semantic search across your Obsidian vault using Ollama's embedding models. It creates vector embeddings of your notes and allows you to search for content based on meaning rather than just keywords.

## Prerequisites

Before using this plugin, you need to have:

1. **Ollama** installed and running on your machine
   - Download from [Ollama's website](https://ollama.ai/)
   - Make sure the Ollama service is running (default port: 11434)

2. **An embedding model** pulled into Ollama
   - The default model is `jeffh/intfloat-multilingual-e5-large-instruct:q8_0`
   - You can use other embedding models like `nomic-embed-text` or any model that supports the embedding API

## Installation

1. Install the plugin from Obsidian's Community Plugins
2. Enable the plugin in Obsidian settings

## Setup

1. Go to the plugin settings in Obsidian
2. Verify the Ollama endpoint (default: `http://localhost:11434`)
3. Choose your preferred embedding model
4. Configure chunk size and overlap settings if needed
5. Click "Build Index" to create the vector index of your notes

## Usage

1. Click the search icon in the ribbon or use the command "Open Ollama Search"
2. Type your query (at least 3 characters)
3. View semantically relevant results from your notes
4. Click on a result to open the corresponding note

## Troubleshooting

If you encounter issues:
- Make sure Ollama is running
- Check that you have pulled the specified embedding model
- Review the error logs in the plugin settings
- Try rebuilding the index

## Performance Considerations

- The initial indexing process may take some time depending on the size of your vault
- Larger embedding models provide better results but may be slower
- Adjust chunk size based on your needs (smaller chunks for more specific results, larger chunks for more context)

## Advanced Configuration

- **Chunk Size**: Controls how your notes are split for embedding (default: 1000 characters)
- **Chunk Overlap**: Determines how much overlap between chunks (default: 200 characters)
- **Max Results**: Sets the maximum number of search results to display (default: 5)

---

This plugin requires a local Ollama installation and does not send your data to external services.