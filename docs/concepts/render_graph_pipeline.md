# Graphified Vector Rendering Pipeline (Skia-shaped)

This document defines the graph-native rendering contract for Thing-OS. Apps publish only a
packed DrawList plus generation counters. Bloom derives and memoizes intermediate artifacts
in the graph using content hashes, so geometry/coverage can be reused when paint changes.

## Canonical layering

- Apps / Clients
  - Publish DrawList bytes + generation counters.
  - No damage, caching, or raster decisions.
- Graph
  - Owns truth: drawlists, assets, window topology.
  - Only IPC path (watch on drawlist gen).
- Bloom
  - Owns work: compute bounds, damage, caches, rasterization, composition.
  - Writes back derived artifacts as optional hints.

## App-published window properties

Required per `UI_WINDOW`:

- `ui.drawlist.bytespace` (`keys::UI_DRAWLIST_BYTESPACE`)
- `ui.drawlist.gen` (`keys::UI_DRAWLIST_GEN`)
- `ui.viewport.bytespace` (`keys::UI_VIEWPORT_BYTESPACE`, `abi::geometry::RectI32Wire`)

Optional:

- `ui.transform.bytespace` (`keys::UI_TRANSFORM_BYTESPACE`, `abi::geometry::Mat3x2fWire`)
- `ui.transform.gen` (`keys::UI_TRANSFORM_GEN`)
- `ui.clip.bytespace` (`keys::UI_CLIP_BYTESPACE`)
- `ui.clip.gen` (`keys::UI_CLIP_GEN`)

Bloom watches only `ui.drawlist.gen` (plus optional transform/clip gens). No per-primitive watches.

## DrawList wire format

See `abi::drawlist` for the packed TLV format. Commands are immutable generations. A drawlist
update must bump `ui.drawlist.gen` and rewrite the drawlist bytespace.

## Renderer-derived node kinds

Renderer owns derived, memoized artifacts, keyed by content hashes:

- `render.Path` (`kinds::RENDER_PATH`)
- `render.FlattenedPath` (`kinds::RENDER_FLATTENED_PATH`)
- `render.EdgeList` (`kinds::RENDER_EDGE_LIST`)
- `render.ScanlineSpans` (`kinds::RENDER_SCANLINE_SPANS`)
- `render.CoverageTile` (`kinds::RENDER_COVERAGE_TILE`)
- `render.Paint` (`kinds::RENDER_PAINT`)
- `render.RasterTile` (`kinds::RENDER_RASTER_TILE`)
- `render.Snapshot` (`kinds::RENDER_SNAPSHOT`)

Each artifact node sets:

- `render.hash` (`keys::RENDER_HASH`)
- `render.schema_version` (`keys::RENDER_SCHEMA_VERSION`)

Bloom can store these under a `render.CacheRoot` node referenced by
`render.cache_root` on the window.

## Hashing and caching rules

Hash definition:

```
hash = H(kind_tag || schema_version || input_hashes || params)
```

Notes:

- Use a stable, deterministic hash (FNV-1a or equivalent).
- Always include `schema_version` so caches can be invalidated.
- Serialize numeric params canonically (LE bytes for integers, IEEE754 bits for floats).

## Invalidation table

- Path geometry changes → invalidate FlattenedPath, EdgeList, Spans, CoverageTile, RasterTile
- Transform changes → invalidate FlattenedPath and downstream (reuse Path)
- Clip changes → invalidate Spans, CoverageTile, RasterTile (reuse Path/EdgeList)
- Paint-only changes → invalidate RasterTile only (reuse Coverage)
- Blend/format changes → invalidate RasterTile only

## Hot loop (Bloom)

1. Drain graph watches (drawlist gen/transform/clip)
2. For each affected window, ingest drawlist bytes
3. Hash -> lookup or create Path / Flattened / Edges
4. Compute coverage tiles for dirty regions
5. Rasterize tiles using cached coverage + paint
6. Composite and present

## Debug/Introspection

Optional per-window debug node `render.Debug.Stats` (`kinds::RENDER_DEBUG_STATS`) with:

- `render.debug.paths`
- `render.debug.flattened`
- `render.debug.edges`
- `render.debug.tiles`
- `render.debug.cache_hits`
- `render.debug.cache_misses`

Bloom may also publish coverage tiles for inspection. These are optional hints only.
