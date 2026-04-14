# Damage Tracking System

## Overview

The Bloom compositor implements a first-class, auditable damage tracking pipeline that makes rendering decisions transparent and debuggable. Every damage rect has an explicit cause, making it possible to answer "Why did this redraw?" at any time.

## Architecture

### DamageCause Enum

Every damage invalidation has an explicit cause:

```rust
enum DamageCause {
    GeometryChanged,  // Window moved or resized
    PaintChanged,     // Style/color properties changed
    AssetUpdated,     // Font, image, or cursor updated
    CursorMoved,      // Cursor position changed
    ForceFull,        // Explicit full redraw requested
    ContentChanged,   // Window snapshot content updated
    FontChanged,      // Font rendering changed
    ThemeChanged,     // Theme settings changed
    Unknown,          // Cause not specified
}
```

Each cause has an associated debug color for visualization:
- **GeometryChanged**: Cyan
- **PaintChanged**: Magenta
- **AssetUpdated**: Yellow
- **CursorMoved**: Green
- **ForceFull**: Red
- **ContentChanged**: Orange
- **FontChanged**: Light Blue
- **ThemeChanged**: Light Pink
- **Unknown**: Gray

### Generation-Based Invalidation

Windows and surfaces track generation counters to prevent accidental redraws:

- `geometry_gen`: Changes when position (x, y) or size (width, height) changes
- `paint_gen`: Changes when visual properties change (uses epoch)
- `asset_gen`: Changes when assets (fonts, images) are updated

The `compute_geometry_generation()` function creates a deterministic hash from geometry parameters, ensuring the generation only changes when actual geometry changes.

### Damage Journal (Debug Only)

In debug builds, Bloom maintains a rolling 60-frame damage history:

```rust
pub struct DamageJournal {
    frames: Vec<FrameDamageRecord>,
    next_frame_id: u64,
}

pub struct FrameDamageRecord {
    frame_id: u64,
    records: Vec<DamageRecord>,
    is_full: bool,
}

pub struct DamageRecord {
    rect: Rect,
    cause: DamageCause,
    source: Option<ThingId>,
}
```

The journal allows post-mortem analysis of damage patterns and can help identify performance issues or unnecessary redraws.

## Deterministic Modes

The compositor supports several deterministic rendering modes for debugging:

### force_full_damage
Always redraws the entire frame, regardless of actual damage. Useful for:
- Validating that damage tracking produces identical output
- Benchmarking full-frame rendering performance
- Debugging rendering artifacts

### disable_damage_tracking
Completely disables damage tracking and always does full redraws. Different from `force_full_damage` in that it bypasses the damage calculation entirely.

### replay_last_frame_damage
Replays the damage from the previous frame. Useful for:
- Comparing consecutive frame damage patterns
- Identifying flicker or redraw issues
- Testing damage accumulation logic

## Debug Visualization

### Keyboard Shortcuts

The compositor provides keyboard shortcuts for toggling various debug modes:

- **F9**: Toggle damage rect overlay (merged rects in magenta)
- **Shift+F9**: Toggle raw damage rect overlay (pre-merge rects in cyan)
- **F8**: Toggle damage cause color-coded overlay
- **F6**: Toggle damage statistics display
- **F5**: Toggle force full damage mode
- **F4**: Toggle disable damage tracking
- **F3**: Toggle replay last frame damage

### Statistics Display

When enabled (F6), the compositor shows:
```
DAMAGE: merged=3 raw=5 reason=[GeometryChanged, CursorMoved]
```

- `merged`: Number of damage rects after consolidation
- `raw`: Number of damage rects before consolidation
- `reason`: List of snapshot invalidation causes
- `(overflow)`: Indicates if rect limit was exceeded

## API Usage

### Recording Damage with Causes

```rust
// Simple damage recording (cause defaults to Unknown)
damage.add_rect(rect);

// Explicit cause and source
damage.add_rect_with_cause(
    rect,
    DamageCause::GeometryChanged,
    Some(window_id),
);

// Full frame damage with cause
damage = Damage::full_with_cause(
    bounds,
    DamageCause::ForceFull,
    None,
);
```

### Using DamageTracker

```rust
let mut tracker = DamageTracker::new();

// Start a new frame
tracker.begin_frame(screen_w, screen_h);

// Record damage with causes
tracker.note_bbox_with_cause(
    window_rect,
    DamageCause::GeometryChanged,
    Some(window_id),
);

// Cursor movement (automatically uses CursorMoved cause)
tracker.note_cursor_move(old_cursor_rect, new_cursor_rect);

// Get accumulated damage
let damage = tracker.end_frame();
```

### Iterating Over Damage Records

```rust
// Get just the rects
for rect in damage.iter() {
    // ...
}

// Get full records (rect + cause + source)
for record in damage.iter_records() {
    println!("Damaged {} because {:?}", record.rect, record.cause);
    if let Some(source) = record.source {
        println!("  Source: {:?}", source);
    }
}
```

## Testing

The damage tracking system has comprehensive unit test coverage:

### Damage Tests
- Cause preservation through rect merging
- Full damage with explicit causes
- Accessor methods (get_cause, get_source)
- Rect consolidation with cause tracking

### DamageTracker Tests
- Cursor movement tracking
- Explicit cause recording
- Full damage marking with cause
- Source tracking

### DamageJournal Tests (Debug Only)
- Frame recording
- Sliding window behavior (60 frame limit)
- Frame ID sequencing

### Generation Tests
- Stability (same input produces same output)
- Position changes (x, y)
- Size changes (width, height)
- Edge cases (negative coords, zero size)

## Performance Considerations

### Release Builds
- All journal functionality is compiled out (`#[cfg(debug_assertions)]`)
- Debug color codes are available in all builds for runtime toggles
- Cause tracking has minimal overhead (just enum and optional ThingId)

### Debug Builds
- Journal maintains last 60 frames
- Uses `Vec` for frame storage (could be optimized to `VecDeque` if needed)
- Removes oldest frame on overflow (O(n) operation, but only happens once per frame)

## Future Enhancements

Potential improvements identified but not yet implemented:

1. **VecDeque for Journal**: Replace `Vec::remove(0)` with `VecDeque::pop_front()` for O(1) sliding window performance

2. **Multi-Cause Tracking**: Currently, when merging rects with different causes, only the first non-Unknown cause is preserved. Could track multiple causes per rect.

3. **Cause Priority**: When merging, could use a priority system to prefer more specific causes over generic ones.

4. **Metrics Export**: Export damage statistics to telemetry/metrics system for production monitoring.

5. **Replay from Journal**: Allow replaying arbitrary frames from the journal, not just the last frame.

## Acceptance Criteria

✅ Every damage rect has a recorded cause  
✅ Live visualization of damage causes  
✅ Deterministic rendering modes for debugging  
✅ Damage decisions are explainable and auditable  
✅ Comprehensive test coverage  
✅ Works in both debug and release builds  

## References

- [Damage Tracking Module](../userspace/bloom/src/damage.rs)
- [Window Snapshot](../userspace/bloom/src/snapshot.rs)
- [Main Compositor Loop](../userspace/bloom/src/main.rs)
- [State Management](../userspace/bloom/src/state.rs)
