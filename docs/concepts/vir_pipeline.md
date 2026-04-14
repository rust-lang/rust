# Vector IR (VIR) Pipeline

## Overview

The Vector IR (VIR) pipeline provides a clean separation between vector asset formats (like SVG) and rendering. This design improves fidelity, enables caching, and makes the rendering deterministic.

## Architecture

```
SVG (or other format)
    ↓
  Parse
    ↓
VIR Document (normalized, high-precision)
    ↓
Tessellation (flatten curves, expand strokes)
    ↓
TessellatedPath (line segments only)
    ↓
DrawList (rasterization commands)
    ↓
Raster (pixels)
```

## Key Principles

1. **Separation of Concerns**: SVG is just an input format. VIR is the canonical representation.
2. **High Precision**: Transforms use f64 until final rasterization to avoid accumulation errors.
3. **Deterministic**: Same input → same output. Useful for caching and testing.
4. **Explicit State**: No inherited attributes. Everything is fully resolved.

## Components

### VIR Types (`vir/types.rs`)

Core data structures:
- `VirPath`: Sequence of path segments (MoveTo, LineTo, QuadTo, CubicTo, Close)
- `VirElement`: A drawable with path, fill, stroke, transform, and opacity
- `VirDocument`: Collection of elements plus metadata
- `Paint`: Currently just solid colors (gradients deferred)
- `FillStyle`: Paint + fill rule (NonZero or EvenOdd)
- `StrokeStyle`: Paint + width + joins + caps + miter limit

### Transform (`vir/transform.rs`)

- Uses f64 internally for precision
- Supports compose, scale, translate, rotate
- Converts to f32 only at final application
- Provides `max_scale()` for tolerance adjustment

### Shape Expansion (`vir/shapes.rs`)

Converts high-level shapes to paths:
- `rect_to_path`: Rectangles (with optional rounded corners)
- `circle_to_path`, `ellipse_to_path`: Approximated with cubic beziers
- `line_to_path`, `polyline_to_path`, `polygon_to_path`

### SVG Conversion (`vir/svg_convert.rs`)

Converts SVG IR to VIR:
- Maps SVG operations to VIR elements
- Resolves colors, transforms, styles
- Preserves fill rules and stroke properties

### Tessellation (`tessellate/`)

Converts curves to line segments:

**Flattening** (`tessellate/flatten.rs`):
- Recursive subdivision for quadratic and cubic beziers
- Tolerance-based (smaller = more segments = higher quality)
- Transform-aware: adjusts tolerance based on scale

**Stroke Expansion** (`tessellate/stroke.rs`):
- Converts stroked paths to filled outlines
- Handles line caps (butt, round, square)
- Handles line joins (miter, round, bevel)
- Miter limit enforcement

### Rendering (`vir/render.rs`)

Bridges VIR to DrawList:
- Tessellates each element
- Converts tessellated vertices to DrawList commands
- Currently uses simple bounding boxes (full polygon rasterization deferred)

## Usage

### Basic Example

```rust
use bloom::vir::*;
use bloom::tessellate::TessellateConfig;

// Create a red circle
let circle_path = shapes::circle_to_path(50.0, 50.0, 25.0);

let element = VirElement {
    path: Arc::new(circle_path),
    fill: Some(FillStyle {
        paint: Paint::Solid(VirColor::rgb(255, 0, 0)),
        rule: FillRule::NonZero,
    }),
    stroke: None,
    transform: VirTransform::identity(),
    opacity: 1.0,
};

let mut doc = VirDocument::new();
doc.elements.push(element);

// Convert to DrawList for rendering
let config = TessellateConfig::default();
let drawlist = vir_to_drawlist(&doc, &config);
```

### SVG to VIR

```rust
use bloom::vir::*;
use bloom::svg::ir::SvgIrDocument;

// Parse SVG (using existing parser)
let svg: SvgIrDocument = /* ... */;

// Convert to VIR
let vir = svg_to_vir(&svg);

// Now render with chosen quality
let config = TessellateConfig {
    tolerance: 0.1, // High quality
    apply_transform: true,
};
let drawlist = vir_to_drawlist(&vir, &config);
```

### Custom Shapes

```rust
let mut path = VirPath::new();
path.move_to(0.0, 0.0);
path.cubic_to(0.0, 50.0, 100.0, 50.0, 100.0, 0.0); // S-curve
path.close();

let element = VirElement {
    path: Arc::new(path),
    fill: Some(FillStyle {
        paint: Paint::Solid(VirColor::rgb(0, 128, 255)),
        rule: FillRule::NonZero,
    }),
    stroke: Some(StrokeStyle {
        paint: Paint::Solid(VirColor::BLACK),
        width: 2.0,
        line_cap: LineCap::Round,
        line_join: LineJoin::Round,
        miter_limit: 4.0,
    }),
    transform: VirTransform::scale(2.0, 2.0),
    opacity: 0.8,
};
```

## Configuration

### Tessellation Quality

```rust
// Default (balanced)
let config = TessellateConfig::default();

// High quality (more segments, smoother curves)
let config = TessellateConfig {
    tolerance: 0.1,
    apply_transform: true,
};

// Low quality (fewer segments, faster)
let config = TessellateConfig {
    tolerance: 1.0,
    apply_transform: true,
};
```

The tolerance is in internal units. It's automatically scaled based on transforms, so a 2x scale will halve the effective tolerance to maintain quality.

## Testing

Run tests:
```bash
cargo test -p bloom vir
```

See `vir/tests.rs` for integration tests and `vir/examples.rs` for usage examples.

## Future Work

### Caching
- Cache VIR by asset hash (avoid re-parsing)
- Cache tessellation by (hash + quality + transform)
- Cache rasters at common sizes

### Coverage AA
- Implement proper coverage-based antialiasing
- Support supersampling (2×2, 4×4)
- Analytic edge coverage (harder but better)

### Full Polygon Rasterization
- Replace bounding box hack with proper triangle fans
- Respect fill rules (non-zero winding vs even-odd)
- Scanline or tile-based rasterization

### Advanced Features
- Gradients (linear, radial)
- Patterns
- Masks and clipping paths
- Text on path
- Dashed strokes

## References

- [SVG Path Specification](https://www.w3.org/TR/SVG/paths.html)
- [Bézier Curve Flattening](https://www.codeproject.com/Articles/32529/Flattening-Bezier-Curves)
- [Stroke Expansion](https://www.microsoft.com/en-us/research/publication/rendering-vector-art-on-the-gpu/)
