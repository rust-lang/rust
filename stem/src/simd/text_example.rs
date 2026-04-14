//! Example demonstrating SIMD-accelerated glyph rendering.
//!
//! This example shows how to:
//! 1. Create a simple atlas with glyph masks
//! 2. Build a positioned glyph run
//! 3. Render text using SIMD compositing
//!
//! To run this example:
//! ```
//! cargo test -p stem --lib simd::text::example --nocapture
//! ```

#[cfg(test)]
mod example {
    use crate::simd::text::*;
    extern crate alloc;
    use alloc::vec;

    #[test]
    fn example_render_text() {
        // Step 1: Create a simple atlas
        // In a real system, this would come from fontd
        let atlas_w = 128;
        let atlas_h = 128;
        let mut atlas = vec![0u8; (atlas_w * atlas_h) as usize];

        // Create simple 8x8 glyphs for "HELLO"
        // Each glyph is a simple pattern for demonstration
        let glyphs = vec![
            ('H', create_h_glyph()),
            ('E', create_e_glyph()),
            ('L', create_l_glyph()),
            ('O', create_o_glyph()),
        ];

        // Pack glyphs into atlas
        let mut placements = alloc::vec::Vec::new();
        let mut atlas_x = 0;
        
        for (glyph_id, (ch, mask)) in glyphs.iter().enumerate() {
            // Copy mask into atlas
            for y in 0..8 {
                for x in 0..8 {
                    let atlas_offset = (y * atlas_w + (atlas_x + x)) as usize;
                    atlas[atlas_offset] = mask[(y * 8 + x) as usize];
                }
            }

            // Create placement (all phases use same rect for this example)
            let rect = Rect::new(atlas_x as i32, 0, 8, 8);
            placements.push(GlyphPlacement {
                glyph_id: *ch as u32,
                phase_rects: [rect; PHASE_COUNT],
                bearing_x: 0,
                bearing_y: 6, // Baseline at y=6
                advance: 9,   // 8px glyph + 1px spacing
            });

            atlas_x += 10; // Next glyph position in atlas
        }

        // Step 2: Build positioned glyph run
        // Render "HELLO" at position (10, 20) with subpixel positions
        let text = "HELLO";
        let mut positioned = alloc::vec::Vec::new();
        let mut x_pos = float_to_subpixel(10.0);
        let y_pos = 20;

        for (i, ch) in text.chars().enumerate() {
            // Add slight subpixel offset to demonstrate phase selection
            let offset = (i as f32 * 0.3) % 1.0; // 0.0, 0.3, 0.6, 0.9, 0.2
            let x_subpixel = float_to_subpixel(subpixel_to_int(x_pos) as f32 + offset);
            
            let frac = subpixel_frac(x_subpixel);
            let phase = compute_phase(frac);

            positioned.push(PositionedGlyph {
                x_subpixel,
                y: y_pos,
                glyph_id: ch as u32,
                phase,
            });

            x_pos += float_to_subpixel(9.0); // Advance by glyph width + spacing
        }

        let run = create_glyph_run(positioned, placements, atlas_w, atlas_h);

        // Step 3: Render to destination surface
        let dst_w = 80;
        let dst_h = 40;
        let mut dst = vec![0xFF000000u32; (dst_w * dst_h) as usize]; // Black background

        let clip = Rect::new(0, 0, dst_w as i32, dst_h as i32);
        let white = 0xFFFFFFFF; // Opaque white

        draw_glyph_run(
            &mut dst,
            dst_w as usize,
            &atlas,
            atlas_w as usize,
            &run,
            &clip,
            white,
        );

        // Step 4: Verify rendering
        // Check that some pixels were drawn (not all black background)
        let white_pixels = dst.iter().filter(|&&p| p == 0xFFFFFFFF).count();
        assert!(white_pixels > 0, "Should have rendered some white pixels");

        // Print ASCII representation for visual verification
        println!("\nRendered 'HELLO':");
        for y in 15..30 {
            for x in 5..60 {
                let px = dst[(y * dst_w + x) as usize];
                let ch = if px == 0xFFFFFFFF {
                    '#'
                } else if px == 0xFF000000 {
                    ' '
                } else {
                    '.' // Partially blended
                };
                print!("{}", ch);
            }
            println!();
        }
        println!();

        // Verify text bounds
        let min_x = 10;
        let max_x = 10 + 5 * 9; // 5 glyphs * 9 width
        let min_y = 20 - 6; // y - bearing_y
        let max_y = 20 + 2; // y + (height - bearing_y)

        // Check that pixels outside bounds are untouched
        assert_eq!(dst[0], 0xFF000000, "Top-left should be background");
        assert_eq!(
            dst[((dst_h - 1) * dst_w + (dst_w - 1)) as usize],
            0xFF000000,
            "Bottom-right should be background"
        );

        println!("✓ Example completed successfully");
        println!("  Rendered {} white pixels", white_pixels);
        println!("  Text bounds: ({}, {}) to ({}, {})", min_x, min_y, max_x, max_y);
    }

    // Helper functions to create simple glyph masks

    fn create_h_glyph() -> alloc::vec::Vec<u8> {
        // 8x8 'H' pattern
        vec![
            255,   0,   0, 255,   0,   0, 255,   0,
            255,   0,   0, 255,   0,   0, 255,   0,
            255,   0,   0, 255,   0,   0, 255,   0,
            255, 255, 255, 255, 255, 255, 255,   0,
            255,   0,   0, 255,   0,   0, 255,   0,
            255,   0,   0, 255,   0,   0, 255,   0,
            255,   0,   0, 255,   0,   0, 255,   0,
              0,   0,   0,   0,   0,   0,   0,   0,
        ]
    }

    fn create_e_glyph() -> alloc::vec::Vec<u8> {
        // 8x8 'E' pattern
        vec![
            255, 255, 255, 255, 255, 255,   0,   0,
            255,   0,   0,   0,   0,   0,   0,   0,
            255,   0,   0,   0,   0,   0,   0,   0,
            255, 255, 255, 255, 255,   0,   0,   0,
            255,   0,   0,   0,   0,   0,   0,   0,
            255,   0,   0,   0,   0,   0,   0,   0,
            255, 255, 255, 255, 255, 255,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,
        ]
    }

    fn create_l_glyph() -> alloc::vec::Vec<u8> {
        // 8x8 'L' pattern
        vec![
            255,   0,   0,   0,   0,   0,   0,   0,
            255,   0,   0,   0,   0,   0,   0,   0,
            255,   0,   0,   0,   0,   0,   0,   0,
            255,   0,   0,   0,   0,   0,   0,   0,
            255,   0,   0,   0,   0,   0,   0,   0,
            255,   0,   0,   0,   0,   0,   0,   0,
            255, 255, 255, 255, 255, 255,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,
        ]
    }

    fn create_o_glyph() -> alloc::vec::Vec<u8> {
        // 8x8 'O' pattern
        vec![
              0, 255, 255, 255, 255, 255,   0,   0,
            255,   0,   0,   0,   0,   0, 255,   0,
            255,   0,   0,   0,   0,   0, 255,   0,
            255,   0,   0,   0,   0,   0, 255,   0,
            255,   0,   0,   0,   0,   0, 255,   0,
            255,   0,   0,   0,   0,   0, 255,   0,
              0, 255, 255, 255, 255, 255,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,
        ]
    }
}
