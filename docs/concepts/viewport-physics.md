# Viewport Physics: How to Use

This guide explains how to use the unified viewport system in Thing-OS for pan, zoom, and scroll interactions.

## Overview

The Viewport Physics system provides a single, reusable camera transform for all canvas-like UIs. It handles:

- **Pan**: Click-drag panning with inertia
- **Zoom**: Mouse wheel zoom, pinch zoom (future), keyboard zoom
- **Scroll**: Smooth kinetic scrolling with friction
- **Coordinate transforms**: Screen ↔ World conversions for hit testing and rendering
- **Two-phase zoom**: Smooth raster scaling during vector re-rendering
- **Snapping**: Optional grid and zoom snapping

## Quick Start

### Basic Setup

```rust
use stem::petals::{Viewport, PanZoomController, ViewportConstraints};

// Create a viewport controller
let mut controller = PanZoomController::new(
    Viewport::new(800.0, 600.0),  // Screen size
    ViewportConstraints {
        min_zoom: 0.1,   // 10%
        max_zoom: 10.0,  // 1000%
        bounds: None,    // Infinite canvas
    },
);
```

### Handling Input

```rust
// Mouse wheel zoom
controller.handle_wheel(mouse_x, mouse_y, wheel_delta);

// Drag panning with timestamps for kinetic scrolling
let current_ns = stem::monotonic_ns();  // or your time source

if pointer_down {
    controller.begin_drag(pointer_x, pointer_y, current_ns);
}
if pointer_moved && controller.is_dragging() {
    controller.update_drag(pointer_x, pointer_y, current_ns);
}
if pointer_up {
    controller.end_drag();  // Applies velocity for kinetic scrolling
}

// Keyboard navigation
controller.handle_arrow_key(up, down, left, right);
controller.handle_keyboard_zoom(zoom_in);
controller.handle_home();  // Reset view
```

### Per-Frame Update

```rust
// In your main loop (call every frame)
let dt = 1.0 / 60.0;  // 60 FPS
let needs_redraw = controller.tick(dt);

if needs_redraw {
    // Viewport changed due to inertia - redraw
}
```

### Coordinate Transforms

```rust
// Hit testing: screen → world
let (world_x, world_y) = controller.viewport.screen_to_world(click_x, click_y);

// Check if click hit your content
if content.contains(world_x, world_y) {
    // Handle click
}

// Rendering: world → screen
let (screen_x, screen_y) = controller.viewport.world_to_screen(node.x, node.y);
draw_node(screen_x, screen_y);
```

### Culling

```rust
// Get visible area for culling
let (x, y, w, h) = controller.viewport.visible_bounds();

// Only render nodes in visible area
for node in nodes {
    if node.x >= x && node.x <= x + w &&
       node.y >= y && node.y <= y + h {
        render_node(node);
    }
}
```

## Advanced Features

### Two-Phase Zoom (Smooth Raster Cache)

For vector graphics that take time to re-render, use `ViewportRenderBridge` to display a scaled raster cache immediately while the vector scene re-renders:

```rust
use stem::petals::viewport::{ViewportRenderBridge, RasterHandle};

// Create bridge
let mut bridge = ViewportRenderBridge::new(
    "my_scene".into(),
    controller.viewport,
);

// When viewport changes
if controller.tick(dt) {
    let current_ns = stem::monotonic_ns();
    
    if bridge.update_target(controller.viewport, current_ns) {
        // Viewport changed significantly - request reraster
        bridge.start_render();
        // ... trigger async vector render ...
    }
}

// Rendering
if bridge.is_using_cache() {
    // Draw scaled raster cache for instant feedback
    if let Some((scale_x, scale_y, tx, ty)) = bridge.get_cache_transform() {
        draw_scaled_raster(bridge.last_raster, scale_x, scale_y, tx, ty);
    }
} else {
    // Draw vector scene directly
    draw_vector_scene(controller.viewport);
}

// When vector render completes
bridge.complete_render(new_raster_handle, controller.viewport);
```

### Inertia Configuration

```rust
// Customize inertia physics
controller.inertia.pan_friction = 0.95;  // Less friction (smoother coast)
controller.inertia.zoom_friction = 0.88;  // Default

// Stop all inertia immediately (e.g., on new user input)
controller.inertia.stop();
```

### Snapping

```rust
use stem::petals::viewport::{snap_zoom, snap_pan, ZoomSnapPolicy, PanSnapPolicy};

// Snap zoom to powers of two
let snapped_zoom = snap_zoom(
    controller.viewport.zoom,
    ZoomSnapPolicy::PowersOfTwo,
    0.05  // 5% threshold
);
controller.viewport.zoom = snapped_zoom;

// Snap pan to grid
let snapped_pos = snap_pan(
    controller.viewport.center_world,
    PanSnapPolicy::Grid { cell_size: 50.0 },
    0.1  // 10% threshold
);
controller.viewport.center_world = snapped_pos;
```

## Integration Examples

### Example: Simple Graph Viewer

```rust
use stem::petals::{Viewport, PanZoomController, ViewportConstraints};

struct GraphViewer {
    controller: PanZoomController,
    nodes: Vec<Node>,
}

impl GraphViewer {
    fn new(width: f32, height: f32) -> Self {
        Self {
            controller: PanZoomController::new(
                Viewport::new(width, height),
                ViewportConstraints::default(),
            ),
            nodes: Vec::new(),
        }
    }

    fn handle_pointer_down(&mut self, x: f32, y: f32) {
        let timestamp = stem::monotonic_ns();
        self.controller.begin_drag(x, y, timestamp);
    }

    fn handle_pointer_move(&mut self, x: f32, y: f32) {
        let timestamp = stem::monotonic_ns();
        if self.controller.update_drag(x, y, timestamp) {
            // Dragging - viewport updated
        } else {
            // Hovering - check hit test
            let (wx, wy) = self.controller.viewport.screen_to_world(x, y);
            // ... check if hovering over node ...
        }
    }

    fn handle_pointer_up(&mut self, x: f32, y: f32) {
        if !self.controller.is_dragging() {
            // Was a click, not a drag
            let (wx, wy) = self.controller.viewport.screen_to_world(x, y);
            self.select_node_at(wx, wy);
        }
        self.controller.end_drag();
    }

    fn handle_wheel(&mut self, x: f32, y: f32, delta: f32) {
        self.controller.handle_wheel(x, y, delta);
    }

    fn tick(&mut self, dt: f32) -> bool {
        self.controller.tick(dt)
    }

    fn render(&self) {
        // Get visible bounds for culling
        let (vx, vy, vw, vh) = self.controller.viewport.visible_bounds();

        // Render only visible nodes
        for node in &self.nodes {
            if node.x >= vx && node.x <= vx + vw &&
               node.y >= vy && node.y <= vy + vh {
                let (sx, sy) = self.controller.viewport.world_to_screen(node.x, node.y);
                draw_node(sx, sy, node);
            }
        }
    }
}
```

### Example: SVG Editor with Smooth Zoom

```rust
use stem::petals::viewport::{ViewportRenderBridge, RasterHandle};

struct SvgEditor {
    controller: PanZoomController,
    bridge: ViewportRenderBridge,
    svg_doc: SvgDocument,
}

impl SvgEditor {
    fn new(width: f32, height: f32) -> Self {
        let viewport = Viewport::new(width, height);
        Self {
            controller: PanZoomController::new(viewport, ViewportConstraints::default()),
            bridge: ViewportRenderBridge::new("svg_editor".into(), viewport),
            svg_doc: SvgDocument::new(),
        }
    }

    fn tick(&mut self, dt: f32) {
        if self.controller.tick(dt) {
            let current_ns = stem::monotonic_ns();
            
            // Check if we should reraster
            if self.bridge.update_target(self.controller.viewport, current_ns) {
                self.start_vector_render();
            }
        }

        // Also check for debounced reraster (after input settles)
        if self.bridge.should_debounced_reraster(stem::monotonic_ns()) {
            self.start_vector_render();
        }
    }

    fn start_vector_render(&mut self) {
        self.bridge.start_render();
        // ... queue async render job ...
    }

    fn on_render_complete(&mut self, raster: RasterHandle) {
        self.bridge.complete_render(raster, self.controller.viewport);
    }

    fn render(&self) {
        if self.bridge.is_using_cache() {
            // Show scaled cache for smooth zoom
            if let Some((sx, sy, tx, ty)) = self.bridge.get_cache_transform() {
                draw_cached_raster(self.bridge.last_raster, sx, sy, tx, ty);
            }
        } else {
            // Render vector at native resolution
            render_svg(&self.svg_doc, &self.controller.viewport);
        }
    }
}
```

## API Reference

### Core Types

- **`Viewport`**: Camera transform (zoom, center, screen size)
- **`PanZoomController`**: Input handling and constraint enforcement
- **`ViewportConstraints`**: Min/max zoom, optional world bounds
- **`InertiaState`**: Kinetic scrolling physics
- **`ViewportRenderBridge`**: Two-phase zoom with raster cache

### Key Methods

#### Viewport
- `screen_to_world(x, y) -> (wx, wy)`: Screen → world coordinates
- `world_to_screen(wx, wy) -> (x, y)`: World → screen coordinates
- `zoom_about(ax, ay, factor)`: Zoom about anchor point
- `pan_by_screen(dx, dy)`: Pan by screen pixels
- `visible_bounds() -> (x, y, w, h)`: Get visible world rect

#### PanZoomController
- `new(viewport, constraints)`: Create controller
- `tick(dt) -> bool`: Apply inertia, returns true if updated
- `handle_wheel(x, y, delta)`: Mouse wheel input
- `begin_drag(x, y, timestamp_ns)`: Start drag gesture with timestamp
- `update_drag(x, y, timestamp_ns) -> bool`: Update drag with timestamp, returns true if dragging
- `end_drag()`: End drag gesture, applies velocity for kinetic scrolling
- `handle_arrow_key(up, down, left, right)`: Keyboard pan
- `handle_keyboard_zoom(zoom_in)`: Keyboard zoom
- `handle_home()`: Reset view

#### ViewportRenderBridge
- `new(scene_id, viewport)`: Create bridge
- `update_target(viewport, ns) -> bool`: Update target viewport
- `should_reraster() -> bool`: Check if reraster needed
- `should_debounced_reraster(ns) -> bool`: Check debounced reraster
- `start_render()`: Mark render started
- `complete_render(handle, viewport)`: Mark render complete
- `get_cache_transform() -> Option<(sx, sy, tx, ty)>`: Get cache transform
- `is_using_cache() -> bool`: Check if using scaled cache

## Best Practices

1. **Always call `tick()` every frame** to apply inertia physics
2. **Use `visible_bounds()` for culling** to avoid rendering offscreen content
3. **Stop inertia on new user input** to feel responsive
4. **Use two-phase zoom for expensive renders** (vector graphics, large datasets)
5. **Debounce expensive operations** using `should_debounced_reraster()`
6. **Keep constraints reasonable** (e.g., `min_zoom: 0.1, max_zoom: 10.0`)

## Performance Tips

- Cull offscreen content using `visible_bounds()`
- Use two-phase zoom for renders > 16ms
- Quantize zoom/pan to reduce reraster frequency
- Cache static content at different zoom levels
- Use spatial indexing (quadtree) with visible bounds

## Common Patterns

### Distinguish Click from Drag

```rust
const DRAG_THRESHOLD_PX: f32 = 5.0;

let drag_distance = {
    let dx = current_x - start_x;
    let dy = current_y - start_y;
    libm::sqrtf(dx * dx + dy * dy)
};
let is_click = drag_distance < DRAG_THRESHOLD_PX;

if pointer_up {
    if is_click {
        handle_click(start_x, start_y);
    }
    controller.end_drag();
}
```

### Zoom to Fit Content

```rust
// Fit all content in view with padding
let (min_x, min_y, max_x, max_y) = calculate_bounds(&nodes);
let w = max_x - min_x;
let h = max_y - min_y;
controller.viewport.fit_rect(min_x, min_y, w, h, 50.0);  // 50px padding
```

### Responsive Feel

```rust
// Stop inertia when user interacts
fn on_user_input(&mut self) {
    self.controller.inertia.stop();
}
```

## Troubleshooting

**Problem**: Zoom feels too fast/slow  
**Solution**: Adjust wheel sensitivity in `handle_wheel()` or change `KEYBOARD_ZOOM_FACTOR`

**Problem**: Inertia doesn't stop  
**Solution**: Check friction values (0.9-0.95 typical) and stop threshold

**Problem**: Hit testing is off  
**Solution**: Ensure `screen_to_world()` uses the same viewport as rendering

**Problem**: Zoom doesn't stay under cursor  
**Solution**: Use `zoom_about()` with cursor position, not pan + zoom separately

**Problem**: Rerasters too frequently  
**Solution**: Increase `quantization.zoom_threshold_log2` or `debounce_ms`

## See Also

- `stem/src/petals/viewport/` - Source code
- `userspace/photosynthesis/` - Example usage (system graph viewer)
- Acceptance tests in `stem/src/petals/viewport/*/tests.rs`
