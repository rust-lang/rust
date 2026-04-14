# Masked Compositor Migration

This document summarizes the migration from ad-hoc A8 mask blending loops to the unified SIMD masked compositor API.

## Goal

Consolidate all "A8 mask over" composition through:
- `stem::simd::composite_solid_masked_over` - For solid color with alpha mask
- `stem::simd::composite_src_masked_over` - For source image with alpha mask

## Changes Made

### 1. Text Rendering - Glyph Atlas Blits

**File**: `stem/src/simd/text.rs`

**Before**: Manual pixel-by-pixel blending fallback for edge cases (lines 256-297)
```rust
// Last row case: manually composite since we can't satisfy the assertion
for x in 0..clipped.w as usize {
    let mask_val = atlas_mask[mask_row_offset + x];
    // ... manual premultiplied blend
}
```

**After**: Always use SIMD compositor with adjusted stride parameter
```rust
// Always use SIMD compositor for consistency and performance
crate::perf::counter("simd.text.composite_masked.count", 1);
composite_solid_masked_over(
    &mut dst[dst_row_offset..],
    dst_stride.max(clipped.w as usize), // Ensure stride >= rect_w
    &atlas_mask[mask_row_offset..],
    atlas_stride.max(clipped.w as usize),
    clipped.w as usize,
    1, // Single row
    color_premul,
);
```

**Performance Counters**:
- `simd.text.glyph_run.count` - Number of glyph runs rendered
- `simd.text.composite_masked.count` - Number of masked composite calls

### 2. Vector Rasterizer - Path Fill with AA

**File**: `userspace/bloom/src/raster.rs::fill_path_aa()`

**Before**: Double-nested loop over coverage array with per-pixel blending
```rust
for py in p_start..p_end {
    let iy = (py - clip.y()) as usize;
    for px in px_start..px_end {
        let ix = (px - clip.x()) as usize;
        let cov = coverage[iy * (clip_w as usize) + ix];
        if cov == 0 { continue; }
        let alpha = ((sa as u16 * cov as u16) / SUPERSAMPLE_SAMPLES as u16) as u8;
        blend_pixel(surface, px, py, sr, sg, sb, alpha);
    }
}
```

**After**: Row-by-row SIMD masked compositor with scaled coverage
```rust
for py in p_start..p_end {
    // Scale coverage to create mask for this row
    let row_coverage = &coverage[row_start_idx..row_end_idx];
    let mut row_mask = vec![0u8; row_width];
    
    for (i, &cov) in row_coverage.iter().enumerate() {
        // Scale coverage to 0-255 range
        row_mask[i] = ((cov as u16 * 255) / SUPERSAMPLE_SAMPLES as u16) as u8;
    }
    
    // Call SIMD masked compositor
    stem::simd::composite_solid_masked_over(
        dst_slice, stride, &row_mask, row_width, row_width, 1, color_premul
    );
}
```

**Performance Counter**: `raster.fill_path_aa.simd.count`

### 3. Fallback Text Rasterization

**File**: `userspace/bloom/src/raster.rs::rasterize_text_fallback()`

**Before**: Double-nested loop over fontdue glyph bitmap
```rust
for r in 0..m.height {
    for c in 0..m.width {
        let a = b[r * (m.width as usize) + c];
        if a > 0 {
            blend_pixel(surface, cx, cy, sr, sg, sb, scale_ch(a, sa) as u8);
        }
    }
}
```

**After**: Row-by-row SIMD compositor with glyph bitmap as mask
```rust
for r in 0..m.height {
    // Get mask row from glyph bitmap
    let mask_row = &b[mask_row_start..mask_row_end];
    
    // Call SIMD masked compositor
    stem::simd::composite_solid_masked_over(
        dst_slice, stride, mask_row, row_width, row_width, 1, color_premul
    );
}
```

**Performance Counter**: `raster.text_fallback.simd.count`

### 4. Cleanup

**Removed**:
- Unused helper functions `scale_ch()` and `blend_channel()` from `stem/src/simd/text.rs`
- These are now only used in the scalar SIMD implementation where they belong

## Remaining `blend_pixel()` Uses

The following uses of `blend_pixel()` were **intentionally not converted** because they don't fit the masked compositor pattern or wouldn't benefit from SIMD:

### Geometric Primitives with Sparse Coverage
- **`fill_circle_blend()`** - Circle coverage computed per-pixel via distance formula
- **`fill_arc_clipped_blend()`** - Arc coverage with angular constraints
- **`fill_rect_linear_gradient()`** - Gradient with per-pixel interpolated color/alpha

**Why not converted**: Coverage patterns are not contiguous in memory and each pixel has unique alpha/color computed on-the-fly.

### Scaled/Transformed Image Blitting
- **`blit_alpha()`** (lines 1186-1216) - Transformed image sampling

**Why not converted**: Each destination pixel samples from potentially different source coordinates due to scale/transform. No contiguous mask exists.

### Small Fill Operations
- **`fill_rect_blend()`** - Simple rectangular fills

**Why not converted**: Small rectangles don't benefit from mask buffer allocation overhead. The double-nested loop is adequate for this use case.

### Edge Anti-Aliasing (Non-AA Path Rasterization)
- Various scanline rasterization paths in `fill_path()` without coverage

**Why not converted**: These are aliased (non-AA) paths without coverage masks.

## Performance Impact

### Expected Improvements
1. **Text rendering**: 2-4x speedup on x86_64 with SSE2, minimal change on scalar
2. **Vector path fills**: Row-wise SIMD processing replaces pixel-by-pixel loops
3. **Consistency**: All AA mask operations now use the same well-tested SIMD path

### Counters to Monitor
```
simd.text.glyph_run.count         # Glyph runs rendered
simd.text.composite_masked.count  # Masked composites in text
raster.fill_path_aa.simd.count    # Vector path fills with AA
raster.text_fallback.simd.count   # Fontdue fallback renders
```

## Validation

### Tests Passing
- All 54 stem tests passing, including:
  - `simd::tests::test_solid_masked_simd_correctness`
  - `simd::tests::test_sse2_solid_masked_correctness`
  - `simd::text::tests::test_draw_single_glyph`
  - `simd::text::tests::test_golden_text_render`

### Build Verification
- Kernel builds successfully for x86_64
- No compilation errors or warnings

## Future Work

1. **NEON optimization**: The NEON backend currently falls back to scalar. Could be optimized.
2. **Batch API**: Consider adding a batched API to reduce per-call overhead for small glyphs.
3. **Cache-friendly patterns**: Explore tile-based processing for better cache utilization.

## References

- Issue: "Use the New Masked Compositor Everywhere It Belongs"
- SIMD implementation: `stem/src/simd/{scalar,x86,neon}.rs`
- Text rendering: `stem/src/simd/text.rs`
- Rasterizer: `userspace/bloom/src/raster.rs`
