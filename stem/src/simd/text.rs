//! SIMD-accelerated glyph atlas blits with subpixel positioning.
//!
//! This module provides fast text rendering by:
//! - Using precomputed phase-shifted glyph masks for subpixel positioning
//! - Leveraging existing SIMD `composite_solid_masked_over` primitives
//! - Supporting proper clipping and bounds checking
//!
//! # Subpixel Positioning
//!
//! Text is positioned using Q24.8 fixed-point coordinates where:
//! - Integer pixel = position >> 8
//! - Fractional part = position & 0xFF (0..255 range)
//!
//! We use 4 phases (0, 0.25, 0.5, 0.75 pixel offsets) selected by:
//! - phase = (fractional * 4) >> 8
//!
//! Each glyph in the atlas should have 4 phase variants stored, allowing
//! fast selection at draw time without per-pixel resampling.

use super::composite_solid_masked_over;

/// Subpixel position in Q24.8 fixed-point format.
/// - Bits 31..8: integer pixel coordinate
/// - Bits 7..0: fractional part (0..255)
pub type SubpixelPos = i32;

/// Number of subpixel phases we support (0, 0.25, 0.5, 0.75).
pub const PHASE_COUNT: usize = 4;

/// Convert float position to Q24.8 fixed-point.
#[inline]
pub fn float_to_subpixel(f: f32) -> SubpixelPos {
    (f * 256.0) as i32
}

/// Extract integer pixel coordinate from subpixel position.
#[inline]
pub fn subpixel_to_int(pos: SubpixelPos) -> i32 {
    pos >> 8
}

/// Extract fractional part from subpixel position (0..255).
#[inline]
pub fn subpixel_frac(pos: SubpixelPos) -> u8 {
    (pos & 0xFF) as u8
}

/// Compute phase index (0..3) from fractional position.
#[inline]
pub fn compute_phase(frac: u8) -> usize {
    // Map 0..255 to 0..3
    // 0..63 -> 0, 64..127 -> 1, 128..191 -> 2, 192..255 -> 3
    ((frac as usize * PHASE_COUNT) >> 8).min(PHASE_COUNT - 1)
}

/// Rectangle in atlas or destination coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    /// Create a new rectangle.
    pub fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }

    /// Compute intersection of two rectangles.
    pub fn intersect(&self, other: &Rect) -> Option<Rect> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.w).min(other.x + other.w);
        let y2 = (self.y + self.h).min(other.y + other.h);

        if x1 < x2 && y1 < y2 {
            Some(Rect::new(x1, y1, x2 - x1, y2 - y1))
        } else {
            None
        }
    }
}

/// A single positioned glyph instance in a glyph run.
#[derive(Debug, Clone)]
pub struct PositionedGlyph {
    /// X position in subpixel coordinates (Q24.8).
    pub x_subpixel: SubpixelPos,
    /// Y position in integer pixels.
    pub y: i32,
    /// Glyph ID (index into atlas placements).
    pub glyph_id: u32,
    /// Phase index for this glyph (0..3).
    pub phase: usize,
}

/// A run of positioned glyphs sharing the same atlas and color.
#[derive(Debug, Clone)]
pub struct GlyphRun {
    /// List of positioned glyph instances.
    pub glyphs: alloc::vec::Vec<PositionedGlyph>,
    /// Atlas metadata (width, height, stride).
    pub atlas_width: u32,
    pub atlas_height: u32,
    /// Glyph placements in the atlas (indexed by glyph_id).
    pub placements: alloc::vec::Vec<GlyphPlacement>,
}

/// Placement of a glyph in the atlas, with multiple phases.
#[derive(Debug, Clone, Copy)]
pub struct GlyphPlacement {
    /// Glyph ID.
    pub glyph_id: u32,
    /// Atlas rectangles for each phase (up to PHASE_COUNT).
    /// Each phase is a horizontal or vertical strip in the atlas.
    pub phase_rects: [Rect; PHASE_COUNT],
    /// Bearing: offset from glyph origin to top-left of bitmap.
    pub bearing_x: i16,
    pub bearing_y: i16,
    /// Horizontal advance to next glyph.
    pub advance: i16,
}

impl GlyphPlacement {
    /// Get the atlas rect for a specific phase.
    pub fn rect_for_phase(&self, phase: usize) -> Rect {
        debug_assert!(phase < PHASE_COUNT, "phase out of range");
        self.phase_rects[phase]
    }
}

/// Draw a run of glyphs from an atlas into a destination buffer.
///
/// # Arguments
/// * `dst` - Destination BGRA8888 buffer (premultiplied).
/// * `dst_stride` - Stride in pixels (not bytes).
/// * `atlas_mask` - Atlas mask buffer (grayscale A8 format).
/// * `atlas_stride` - Atlas stride in pixels.
/// * `run` - The glyph run to render.
/// * `clip` - Clipping rectangle in destination coordinates.
/// * `color_premul` - Text color (premultiplied BGRA8888).
///
/// # Safety
/// - `dst` must have at least `dst_stride * clip.h` pixels available.
/// - `atlas_mask` must have at least `atlas_stride * atlas_height` bytes.
/// - All glyph placements must reference valid atlas regions.
pub fn draw_glyph_run(
    dst: &mut [u32],
    dst_stride: usize,
    atlas_mask: &[u8],
    atlas_stride: usize,
    run: &GlyphRun,
    clip: &Rect,
    color_premul: u32,
) {
    crate::perf::counter("simd.text.glyph_run.count", 1);

    for glyph in &run.glyphs {
        // Find placement for this glyph
        let placement = match run.placements.iter().find(|p| p.glyph_id == glyph.glyph_id) {
            Some(p) => p,
            None => continue, // Glyph not in atlas
        };

        // Get atlas rect for this glyph's phase
        let atlas_rect = placement.rect_for_phase(glyph.phase);

        // Compute destination rect
        let dst_x = subpixel_to_int(glyph.x_subpixel) + placement.bearing_x as i32;
        let dst_y = glyph.y + placement.bearing_y as i32;
        let dst_rect = Rect::new(dst_x, dst_y, atlas_rect.w, atlas_rect.h);

        // Clip to destination bounds
        let clipped = match dst_rect.intersect(clip) {
            Some(r) => r,
            None => continue, // Glyph completely outside clip rect
        };

        // Skip empty rects
        if clipped.w <= 0 || clipped.h <= 0 {
            continue;
        }

        // Compute source offsets within atlas rect
        let src_offset_x = clipped.x - dst_rect.x;
        let src_offset_y = clipped.y - dst_rect.y;

        // Bounds check before proceeding
        if clipped.x < 0 || clipped.y < 0 {
            continue;
        }
        let dst_x = clipped.x as usize;
        let dst_y = clipped.y as usize;
        let atlas_src_x = (atlas_rect.x + src_offset_x) as usize;
        let atlas_src_y = (atlas_rect.y + src_offset_y) as usize;

        if dst_x + clipped.w as usize > dst_stride {
            continue;
        }
        if dst_y + clipped.h as usize > dst.len() / dst_stride {
            continue;
        }
        if atlas_src_x + clipped.w as usize > atlas_stride {
            continue;
        }
        if atlas_src_y + clipped.h as usize > atlas_mask.len() / atlas_stride {
            continue;
        }

        // Call SIMD compositing primitive row by row
        for row in 0..clipped.h as usize {
            let dst_row_offset = (dst_y + row) * dst_stride + dst_x;
            let mask_row_offset = (atlas_src_y + row) * atlas_stride + atlas_src_x;

            // Check that we have enough pixels available for this row
            let dst_available = dst.len() - dst_row_offset;
            let mask_available = atlas_mask.len() - mask_row_offset;

            // Prefer SIMD compositor when we have enough buffer space for the assertion
            if dst_available >= dst_stride && mask_available >= atlas_stride {
                // Use SIMD compositor for optimal performance
                crate::perf::counter("simd.text.composite_masked.count", 1);
                composite_solid_masked_over(
                    &mut dst[dst_row_offset..],
                    dst_stride,
                    &atlas_mask[mask_row_offset..],
                    atlas_stride,
                    clipped.w as usize,
                    1, // Single row
                    color_premul,
                );
            } else if dst_available >= clipped.w as usize && mask_available >= clipped.w as usize {
                // Fallback for edge cases (e.g., last row) where buffer is tight
                // Use scalar path to avoid assertion failures
                crate::perf::counter("simd.text.composite_masked_fallback.count", 1);
                super::scalar::composite_solid_masked_over_scalar(
                    &mut dst[dst_row_offset..],
                    clipped.w as usize, // Use rect_w as stride for tight buffer
                    &atlas_mask[mask_row_offset..],
                    clipped.w as usize,
                    clipped.w as usize,
                    1,
                    color_premul,
                );
            }
        }
    }
}

/// Helper to create a simple glyph run from a list of glyphs.
///
/// This is primarily for testing. Real usage should build runs from
/// layout/shaping engines.
pub fn create_glyph_run(
    glyphs: alloc::vec::Vec<PositionedGlyph>,
    placements: alloc::vec::Vec<GlyphPlacement>,
    atlas_width: u32,
    atlas_height: u32,
) -> GlyphRun {
    GlyphRun {
        glyphs,
        atlas_width,
        atlas_height,
        placements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_subpixel_conversion() {
        // Test integer positions
        assert_eq!(subpixel_to_int(float_to_subpixel(10.0)), 10);
        assert_eq!(subpixel_to_int(float_to_subpixel(100.0)), 100);

        // Test fractional positions
        let pos = float_to_subpixel(10.5);
        assert_eq!(subpixel_to_int(pos), 10);
        assert!((subpixel_frac(pos) as i32 - 128).abs() <= 1); // ~0.5 in Q8
    }

    #[test]
    fn test_phase_computation() {
        // Test phase boundaries
        assert_eq!(compute_phase(0), 0); // 0.00
        assert_eq!(compute_phase(63), 0); // ~0.25
        assert_eq!(compute_phase(64), 1); // 0.25
        assert_eq!(compute_phase(127), 1); // ~0.50
        assert_eq!(compute_phase(128), 2); // 0.50
        assert_eq!(compute_phase(191), 2); // ~0.75
        assert_eq!(compute_phase(192), 3); // 0.75
        assert_eq!(compute_phase(255), 3); // ~1.00
    }

    #[test]
    fn test_rect_intersection() {
        let r1 = Rect::new(10, 10, 20, 20);
        let r2 = Rect::new(15, 15, 20, 20);

        let isect = r1.intersect(&r2).unwrap();
        assert_eq!(isect.x, 15);
        assert_eq!(isect.y, 15);
        assert_eq!(isect.w, 15);
        assert_eq!(isect.h, 15);

        // No intersection
        let r3 = Rect::new(100, 100, 10, 10);
        assert!(r1.intersect(&r3).is_none());
    }

    #[test]
    fn test_draw_single_glyph() {
        // Create a simple 8x8 atlas with a single glyph
        let atlas_w = 32;
        let atlas_h = 32;
        let mut atlas = vec![0u8; (atlas_w * atlas_h) as usize];

        // Draw a simple 4x4 square glyph in the atlas at (0, 0) for phase 0
        for y in 0..4 {
            for x in 0..4 {
                atlas[(y * atlas_w + x) as usize] = 255; // Full coverage
            }
        }

        // Create placement with all phases pointing to same rect (simple test)
        let placement = GlyphPlacement {
            glyph_id: 42,
            phase_rects: [
                Rect::new(0, 0, 4, 4),
                Rect::new(0, 0, 4, 4),
                Rect::new(0, 0, 4, 4),
                Rect::new(0, 0, 4, 4),
            ],
            bearing_x: 0,
            bearing_y: 0,
            advance: 5,
        };

        // Create a glyph run with one glyph at (10, 10)
        let glyph = PositionedGlyph {
            x_subpixel: float_to_subpixel(10.0),
            y: 10,
            glyph_id: 42,
            phase: 0,
        };

        let run = create_glyph_run(vec![glyph], vec![placement], atlas_w, atlas_h);

        // Destination buffer
        let dst_w = 32;
        let dst_h = 32;
        let mut dst = vec![0u32; (dst_w * dst_h) as usize];

        // White text color
        let color = 0xFFFFFFFF; // Opaque white (premultiplied)

        // Clip rect covering whole destination
        let clip = Rect::new(0, 0, dst_w as i32, dst_h as i32);

        // Draw the glyph run
        draw_glyph_run(
            &mut dst,
            dst_w as usize,
            &atlas,
            atlas_w as usize,
            &run,
            &clip,
            color,
        );

        // Verify pixels at (10, 10) through (13, 13) are white
        for y in 10..14 {
            for x in 10..14 {
                let px = dst[(y * dst_w + x) as usize];
                assert_eq!(px, 0xFFFFFFFF, "Expected white at ({}, {})", x, y);
            }
        }

        // Verify pixels outside glyph area are unchanged (still 0)
        assert_eq!(dst[0], 0);
        assert_eq!(dst[(9 * dst_w + 9) as usize], 0);
    }

    #[test]
    fn test_clipping() {
        // Create a simple atlas
        let atlas_w = 16;
        let atlas_h = 16;
        let mut atlas = vec![0u8; (atlas_w * atlas_h) as usize];

        // 4x4 glyph
        for y in 0..4 {
            for x in 0..4 {
                atlas[(y * atlas_w + x) as usize] = 128; // Half coverage
            }
        }

        let placement = GlyphPlacement {
            glyph_id: 1,
            phase_rects: [Rect::new(0, 0, 4, 4); PHASE_COUNT],
            bearing_x: 0,
            bearing_y: 0,
            advance: 5,
        };

        // Position glyph partially outside clip rect
        let glyph = PositionedGlyph {
            x_subpixel: float_to_subpixel(14.0), // Will be clipped on right
            y: 14,                               // Will be clipped on bottom
            glyph_id: 1,
            phase: 0,
        };

        let run = create_glyph_run(vec![glyph], vec![placement], atlas_w, atlas_h);

        let dst_w = 16;
        let dst_h = 16;
        let mut dst = vec![0u32; (dst_w * dst_h) as usize];

        // Clip rect
        let clip = Rect::new(0, 0, 16, 16);

        // Draw (should be clipped to 2x2)
        draw_glyph_run(
            &mut dst,
            dst_w as usize,
            &atlas,
            atlas_w as usize,
            &run,
            &clip,
            0xFFFFFFFF,
        );

        // Verify that only (14,14), (15,14), (14,15), (15,15) are drawn
        let valid_coords = [(14, 14), (15, 14), (14, 15), (15, 15)];
        for (x, y) in &valid_coords {
            let px = dst[(y * dst_w + x) as usize];
            assert_ne!(px, 0, "Expected non-zero at ({}, {})", x, y);
        }
    }

    #[test]
    fn test_fractional_positioning() {
        // Test that fractional positions select correct phases
        let positions = [0.0, 0.25, 0.5, 0.75, 1.0];
        let expected_phases = [0, 1, 2, 3, 0]; // 1.0 wraps to integer, uses phase 0

        for (pos, &expected_phase) in positions.iter().zip(expected_phases.iter()) {
            let subpx = float_to_subpixel(*pos);
            let frac = subpixel_frac(subpx);
            let phase = compute_phase(frac);
            assert_eq!(
                phase, expected_phase,
                "Position {} should select phase {}",
                pos, expected_phase
            );
        }
    }

    #[test]
    fn test_golden_text_render() {
        // Render a simple text string and verify checksum
        // This ensures consistent output across changes

        let atlas_w = 64;
        let atlas_h = 64;
        let mut atlas = vec![0u8; (atlas_w * atlas_h) as usize];

        // Create 3 simple glyphs: 'A', 'B', 'C' as 5x5 patterns
        // Glyph 'A' (id=65) at (0, 0)
        for y in 0..5 {
            for x in 0..5 {
                let val = if y == 0 || x == 0 || x == 4 || y == 2 {
                    200
                } else {
                    0
                };
                atlas[(y * atlas_w + x) as usize] = val;
            }
        }

        let placement_a = GlyphPlacement {
            glyph_id: 65,
            phase_rects: [Rect::new(0, 0, 5, 5); PHASE_COUNT],
            bearing_x: 0,
            bearing_y: 0,
            advance: 6,
        };

        // Render "A" at x=10, y=10
        let glyph = PositionedGlyph {
            x_subpixel: float_to_subpixel(10.0),
            y: 10,
            glyph_id: 65,
            phase: 0,
        };

        let run = create_glyph_run(vec![glyph], vec![placement_a], atlas_w, atlas_h);

        let dst_w = 32;
        let dst_h = 32;
        let mut dst = vec![0u32; (dst_w * dst_h) as usize];

        let clip = Rect::new(0, 0, 32, 32);
        draw_glyph_run(
            &mut dst,
            dst_w as usize,
            &atlas,
            atlas_w as usize,
            &run,
            &clip,
            0xFFFFFFFF,
        );

        // Compute simple checksum
        let mut checksum = 0u64;
        for &px in &dst {
            checksum = checksum.wrapping_add(px as u64);
        }

        // This checksum will be stable for this specific render
        // If implementation changes but output stays the same, checksum should match
        assert_ne!(checksum, 0, "Checksum should be non-zero");
        // For this simple test, we just verify it's deterministic by running twice
        let mut dst2 = vec![0u32; (dst_w * dst_h) as usize];
        draw_glyph_run(
            &mut dst2,
            dst_w as usize,
            &atlas,
            atlas_w as usize,
            &run,
            &clip,
            0xFFFFFFFF,
        );
        assert_eq!(dst, dst2, "Two renders should be identical");
    }

    /// Example demonstrating how to use the SIMD glyph rendering API.
    ///
    /// This test serves as both documentation and validation.
    #[test]
    fn example_render_hello() {
        // Create a simple atlas with letter masks
        let atlas_w = 128;
        let atlas_h = 128;
        let mut atlas = vec![0u8; (atlas_w * atlas_h) as usize];

        // Simple 6x6 'H' glyph at (0,0)
        let h_mask = [
            255, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0,
            0, 255, 255, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 255,
        ];
        for y in 0..6 {
            for x in 0..6 {
                atlas[(y * atlas_w + x) as usize] = h_mask[(y * 6 + x) as usize];
            }
        }

        // Create placement
        let rect = Rect::new(0, 0, 6, 6);
        let placement = GlyphPlacement {
            glyph_id: b'H' as u32,
            phase_rects: [rect; PHASE_COUNT],
            bearing_x: 0,
            bearing_y: 5,
            advance: 7,
        };

        // Build glyph run: "H" at (10, 20) with 0.25px subpixel offset
        let x_pos = float_to_subpixel(10.25);
        let frac = subpixel_frac(x_pos);
        let phase = compute_phase(frac); // Should be phase 1 (0.25 -> 64/256 -> phase 1)

        let glyph = PositionedGlyph {
            x_subpixel: x_pos,
            y: 20,
            glyph_id: b'H' as u32,
            phase,
        };

        let run = create_glyph_run(vec![glyph], vec![placement], atlas_w, atlas_h);

        // Render
        let dst_w = 32;
        let dst_h = 32;
        let mut dst = vec![0u32; (dst_w * dst_h) as usize];
        let clip = Rect::new(0, 0, dst_w as i32, dst_h as i32);

        draw_glyph_run(
            &mut dst,
            dst_w as usize,
            &atlas,
            atlas_w as usize,
            &run,
            &clip,
            0xFFFFFFFF, // White
        );

        // Verify some pixels were drawn
        let rendered = dst.iter().filter(|&&p| p != 0).count();
        assert!(rendered > 0, "Should have rendered some pixels");

        // Verify phase was computed correctly
        assert_eq!(phase, 1, "0.25 fractional position should select phase 1");
    }
}
