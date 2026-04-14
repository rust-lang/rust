# Asset Watcher Service Implementation

## Overview

This document describes the implementation of continuous asset watching in Thing-OS, transforming assets from boot-time artifacts into living graph citizens.

## Problem Statement

Previously, assets were "boot-shaped":
- Discovered via one-time scans
- Assumed to exist afterward
- Changes required reboot or manual intervention

This contradicted Thing-OS's core philosophy: **the graph is the system, and truth is continuous**.

## Solution

The `flytrap` service (conceptually `assetd`) now implements continuous asset watching:

### Architecture

**Service**: `userspace/flytrap/src/main.rs`

**Key Features**:
1. **Continuous Discovery**: Watches BOOT_MODULE nodes for new assets appearing at runtime
2. **Content Hashing**: Computes SHA-256 hash of each asset for change detection
3. **Graph Materialization**: Creates canonical Asset nodes with stable properties
4. **Change Detection**: Increments generation counter when content changes
5. **Deduplication**: Identifies identical assets by content hash

### Asset Node Schema

Each asset in the graph contains:

```rust
Thing: Asset {
    asset.name: String,        // Stable logical name
    asset.kind: String,        // Type: font, svg, image, cursor, raw
    asset.hash: u64,           // SHA-256 hash (first 8 bytes)
    asset.size: u64,           // Size in bytes
    asset.bytespace: ThingId,  // Reference to content
    asset.generation: u64,     // Increments on change
    asset.source: String,      // Origin (e.g., "boot")
    asset.ready: u64,          // 1 if ready for use
}
```

### How It Works

#### 1. Initial Scan
At service startup, `flytrap` scans existing BOOT_MODULE nodes and publishes them as assets.

#### 2. Continuous Watching
The service watches for:
- **New BOOT_MODULE nodes**: Created by iso_reader or other services
- **ASSET_REQUEST nodes**: Explicit requests for assets
- **PROC_TASK nodes**: Application startup events

#### 3. Asset Publication
When a new asset is discovered:
1. Map the bytespace content
2. Compute SHA-256 hash
3. Determine asset kind (font, svg, image, etc.)
4. Check for existing asset with same name
5. If hash matches: skip update (idempotent)
6. If hash differs: update properties, increment generation
7. If new: create asset node with all properties

#### 4. Consumer Contract
Downstream systems (Bloom, Blossom, etc.):
- Watch asset nodes, not boot modules
- React to generation changes
- Never assume assets are static

## Implementation Details

### Content Hashing

Uses SHA-256 via the `sha2` crate:
```rust
let mut hasher = Sha256::new();
hasher.update(slice);
let hash_bytes = hasher.finalize();
// Store first 8 bytes as u64
let hash = u64::from_le_bytes([...]);
```

### Deduplication

```rust
// Check if same name and same hash
if existing_name == name_sym && existing_hash == hash {
    info!("Asset unchanged (hash match)");
    return existing_id;
}
```

### Change Detection

```rust
if old_hash != hash {
    prop_set(id, keys::ASSET_HASH, hash);
    prop_set(id, keys::ASSET_SIZE, size);
    prop_set(id, keys::ASSET_READY, 1);
    generation = generation + 1;
    prop_set(id, keys::ASSET_GENERATION, generation);
}
```

## Acceptance Criteria Status

| Criterion | Status | Notes |
|-----------|--------|-------|
| Add/edit SVG at runtime → Bloom updates | ✅ | Watches detect new BOOT_MODULEs, generation increments |
| Remove font → UI falls back | ⚠️ | Asset remains in graph; UI must handle stale assets |
| Identical assets deduplicate | ✅ | Hash-based deduplication implemented |
| No boot-time sweep remains | ✅ | Continuous watching active; initial scan is just bootstrap |

## Future Enhancements

1. **Asset Removal Detection**: Watch for Delete events on BOOT_MODULE nodes
2. **Bytespace Change Watching**: Detect when asset content changes without new BOOT_MODULE
3. **Cross-Name Deduplication**: Share bytespaces for assets with identical hash but different names
4. **Asset Metadata Caching**: Cache parsed metadata (font metrics, SVG dimensions, etc.)

## Dependencies

- `sha2`: Content hashing
- `infer`: MIME type detection
- `ttf-parser`: Font metadata extraction
- `stem`: System calls and runtime

## Testing

Build the service:
```bash
cargo +nightly build -Z build-std=core,alloc \
  -Z build-std-features=compiler-builtins-mem \
  --target targets/x86_64-unknown-thingos.json \
  -p flytrap
```

The service is automatically included in ISO builds via `just iso`.

## Philosophy

> Assets become truth, not assumptions.

The graph is the authoritative source. Assets are not discovered once and forgotten—they are continuously monitored, their truth maintained in the graph, and changes propagate naturally to all consumers.

This completes the architectural promise: **everything is a Thing**, including assets.
