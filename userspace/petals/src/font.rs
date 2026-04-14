#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use crate::Canvas;
use abi::syscall::vfs_flags::O_RDONLY;
use alloc::vec;
use alloc::vec::Vec;
use fontdue::{Font, FontSettings};
use stem::syscall::vfs::{vfs_close, vfs_open, vfs_read, vfs_stat};

pub struct TextRenderer {
    pub font: Font,
}

impl TextRenderer {
    pub fn load_from_boot(path: &str) -> Option<Self> {
        let fd = vfs_open(path, O_RDONLY).ok()?;

        // Get size
        let stat = vfs_stat(fd).ok()?;
        let size = stat.size as usize;

        let mut data = vec![0u8; size];
        if vfs_read(fd, &mut data).ok()? < size {
            let _ = vfs_close(fd);
            return None;
        }
        let _ = vfs_close(fd);

        let font = Font::from_bytes(data, FontSettings::default()).ok()?;
        Some(Self { font })
    }

    pub fn draw_text(
        &self,
        canvas: &mut Canvas,
        text: &str,
        x: i32,
        y: i32,
        px_size: f32,
        color: u32,
    ) {
        let mut cur_x = x as f32;

        for c in text.chars() {
            let (metrics, bitmap) = self.font.rasterize(c, px_size);

            // Blit glyph
            let glyph_x = (cur_x + metrics.xmin as f32) as i32;
            let glyph_y = (y as f32 - metrics.ymin as f32 - metrics.height as f32) as i32;

            for gy in 0..metrics.height {
                let dy = glyph_y + gy as i32;
                if dy < 0 || dy >= canvas.height as i32 {
                    continue;
                }

                for gx in 0..metrics.width {
                    let dx = glyph_x + gx as i32;
                    if dx < 0 || dx >= canvas.width as i32 {
                        continue;
                    }

                    let coverage = bitmap[gy * metrics.width + gx];
                    if coverage > 0 {
                        // Blend color with background
                        let alpha = (((color >> 24) * coverage as u32) / 255) as u8;
                        let effective_color = (color & 0x00FFFFFF) | ((alpha as u32) << 24);

                        let dst_idx = (dy as u32 * canvas.stride_pixels + dx as u32) as usize;
                        let dst_color = canvas.buffer[dst_idx];
                        canvas.buffer[dst_idx] = blend(dst_color, effective_color);
                    }
                }
            }

            cur_x += metrics.advance_width;
        }
    }
}

fn blend(dst: u32, src: u32) -> u32 {
    let alpha = (src >> 24) as u32;
    if alpha == 0 {
        return dst;
    }
    if alpha == 255 {
        return src;
    }

    let inv_alpha = 255 - alpha;

    let sr = (src >> 16) & 0xFF;
    let sg = (src >> 8) & 0xFF;
    let sb = src & 0xFF;

    let dr = (dst >> 16) & 0xFF;
    let dg = (dst >> 8) & 0xFF;
    let db = dst & 0xFF;

    let r = (sr * alpha + dr * inv_alpha) / 255;
    let g = (sg * alpha + dg * inv_alpha) / 255;
    let b = (sb * alpha + db * inv_alpha) / 255;

    (0xFF << 24) | (r << 16) | (g << 8) | b
}
