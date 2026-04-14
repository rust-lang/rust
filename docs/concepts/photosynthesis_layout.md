# Photosynthesis Graph Layout System

## Overview

Photosynthesis uses an incremental force-directed layout algorithm with collision avoidance to visualize the system graph. The layout engine produces non-overlapping node placements, preserves stable positions across updates, and supports smooth incremental re-layout when the graph changes.

## Features

### 1. Incremental Force-Directed Layout

The layout uses a two-stage pipeline:

**Stage A: Coarse Placement (Cluster-Aware)**
- Preserves existing positions for nodes that already have them
- Places new nodes near their neighbors (average of neighbor positions)
- Falls back to spiral placement for isolated nodes
- Uses Golden Angle (Vogel's model) for aesthetically pleasing distribution

**Stage B: Physics Refinement**
- Applies repulsion forces between nodes (avoid overlap)
- Applies attraction forces along edges (keep connected nodes close)
- Resolves rectangle overlaps by pushing nodes apart
- Uses velocity damping for smooth convergence
- Runs bounded iterations per tick (default: 20 iterations max)

### 2. Collision Avoidance

Uses rectangle overlap resolution with minimal separating vector:
- Sweeps through all node pairs
- Detects rectangle overlaps
- Pushes nodes apart along the minimal separating axis
- Respects pinned nodes (only moves the other node)

### 3. Stable Positions

- Layout state persists to the graph via properties:
  - `layout.pos.x`, `layout.pos.y` - world coordinates
  - `layout.size.w`, `layout.size.h` - node size (for collision)
  - `layout.vel.x`, `layout.vel.y` - velocity (for animation)
  - `layout.gen` - generation counter (increments on change)
  - `layout.pin` - user-pinned flag

- Positions are preserved across frames and app restarts
- Only nodes that move significantly trigger property updates

### 4. Node Pinning

Users can click on nodes to pin/unpin them:
- Pinned nodes cannot be moved by the layout algorithm
- Pinned nodes act as anchors, improving mental map stability
- Pin state is persisted to the graph
- Visual indicator shows pinned nodes in debug mode

### 5. Edge Routing

Edges attach to node boundaries instead of centers:
- Calculates intersection point from node center to target
- Uses rectangle edge intersection algorithm
- Draws straight lines with proper attachment points
- Arrow heads indicate edge direction

### 6. Visual Debug Mode

Press 'D' to toggle debug visualization:
- **Red overlay**: Collision boxes (actual node bounds used for collision)
- **Green arrows**: Velocity vectors (showing movement direction and magnitude)
- **Orange dots**: Pin indicators (marks pinned nodes)

## Configuration

Layout behavior is controlled by `LayoutSettings`:

```rust
pub struct LayoutSettings {
    // Grid dimensions
    pub grid_w: f32,              // Cell width (default: 160.0)
    pub grid_h: f32,              // Cell height (default: 120.0)
    
    // Force-directed physics
    pub repulsion_strength: f32,   // Node repulsion (default: 5000.0)
    pub attraction_strength: f32,  // Edge attraction (default: 0.3)
    pub damping: f32,              // Velocity damping (default: 0.85)
    pub min_distance: f32,         // Min node separation (default: 20.0)
    
    // Performance
    pub max_iterations: usize,     // Max iterations/tick (default: 20)
    pub stability_threshold: f32,  // Movement threshold (default: 0.5)
}
```

## Keyboard Shortcuts

- **Arrow keys**: Pan the viewport
- **Alt + Plus/Minus**: Zoom in/out
- **Home**: Reset view
- **Page Up/Down**: Large scroll
- **D**: Toggle debug mode
- **F8**: Cycle locale
- **Click on node**: Pin/unpin node

## Architecture

### Layout Properties Schema

```
layout.pos.x     - X position in world coordinates (f32)
layout.pos.y     - Y position in world coordinates (f32)
layout.size.w    - Node width for collision (f32)
layout.size.h    - Node height for collision (f32)
layout.vel.x     - X velocity (f32, stored as millis)
layout.vel.y     - Y velocity (f32, stored as millis)
layout.pin       - Pin state (bool, 0 or 1)
layout.gen       - Layout generation counter (u64)
```

### Camera Separation

- Layout works in world coordinates
- Viewport/camera handles screen transforms independently
- Pan and zoom are separate from layout

## Performance

- Layout computation is time-budgeted (5ms per tick)
- Early exit when stable (max movement < threshold)
- Incremental updates only re-layout affected regions
- Collision detection uses simple rectangle overlap (O(n²) but fast in practice)

## Future Improvements

Potential enhancements (not required for v0):
- Hierarchical clustering (community detection)
- Edge bundling between clusters
- Orthogonal routing for pipes
- Label collision resolution
- Focus-on-selection layout mode (radial around selected node)
- Spatial indexing for faster collision detection (quadtree/grid)
