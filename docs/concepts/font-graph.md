# Font Graph Model

Thing-OS implements a graph-native font system where fonts are first-class citizens in the Kernel Graph.

## Node Kinds

| Kind | Description |
|------|-------------|
| `font.Superfamily` | Broad grouping (e.g., "Noto") |
| `font.Family` | Collection of related typefaces (e.g., "Noto Sans") |
| `font.Face` | Specific weight/style/stretch (e.g., "Noto Sans Bold") |
| `font.Blob` | Raw font file backing store (bytespace) |
| `font.File` | Metadata about source file |
| `font.Atlas` | Glyph atlas for (face, size) - contains rendered glyphs |
| `font.Coverage` | Character coverage information |
| `font.Glyph` | Individual rasterized glyph (legacy, use Atlas instead) |

## Relationships

| Predicate | From | To | Description |
|-----------|------|-----|-------------|
| `font.contains` | Superfamily/Family | Family/Face | Hierarchy |
| `font.has_face` | Family | Face | Face membership |
| `font.has_blob` | Face | Blob | Raw file backing |
| `font.has_asset` | Face | Bytespace | Legacy alias for has_blob |
| `font.has_atlas` | Face | Atlas | Runtime atlas |
| `font.has_coverage` | Face | Coverage | Character coverage |
| `font.fallback_to` | Family | Family | Fallback chain |

## Required Properties

### Font Family (`font.Family`)
- `font.name` (string bytespace): Human-readable name
- `font.family_key` (u64): Normalized lookup key (interned)

### Font Face (`font.Face`)
- `font.style` (string bytespace): Style name ("Regular", "Bold", etc.)
- `font.weight` (u16): CSS-style weight (100-900, 400=Regular)
- `font.width` (u16): Width/stretch (1-9, 5=Normal)
- `font.slope` (u8): 0=Upright, 1=Italic, 2=Oblique

### Font Blob (`font.Blob`)
- `bytespace` (ThingId): Raw font file data
- `font.blob.sha256` (string): Content hash for identity
- `font.blob.mime` (string): MIME type (`font/ttf`, `font/otf`)

### Font Atlas (`font.Atlas`)
- `font.atlas.bytespace` (ThingId): Pixel buffer
- `font.atlas.width` (u32): Atlas width in pixels
- `font.atlas.height` (u32): Atlas height in pixels
- `font.atlas.format` (u8): 0=A8 (grayscale), 1=RGBA8888
- `font.atlas.version` (u64): Monotonic version for cache invalidation

## Identity Stability Rule

**Critical invariant**: If the same font blob (same SHA256) appears again, it MUST map to the same `FONT_BLOB` ThingId. This is achieved via lookup-or-create keyed by hash.

This ensures:
1. Glyph caches remain valid across restarts
2. Atlas products don't require rebuilding
3. Font references are stable for the session lifetime

## Lifecycle Responsibilities

### FontD (Importer/Curator)
- Scans configured font sources on startup
- Populates/updates font graph nodes
- Owns atlas creation and rasterization
- Serves metrics and glyph requests via IPC

### Bloom (Consumer)
- Queries graph to discover families/faces
- Requests metrics/glyphs from fontd using ThingIds
- Caches atlas mappings locally
- Never modifies font graph nodes
