use crate::requests::FRAMEBUFFER_REQUEST;
use kernel::{FramebufferInfo, PixelFormat};

#[derive(Clone, Copy)]
pub struct Framebuffer {
    pub addr: *mut u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub bpp: u32, // bytes per pixel
}

fn normalize_geometry(width: u32, height: u32, pitch: u32, bpp: u32) -> (u32, u32, u32) {
    let row_bytes = width.saturating_mul(bpp.max(1));
    let mut eff_pitch = if pitch > 0 { pitch } else { row_bytes };
    let mut eff_width = width;
    let eff_height = height;

    // Some firmware/QEMU combos report a width that isn't 8-pixel aligned.
    // The actual scanout often rounds down to an 8-pixel boundary, which
    // effectively reduces the stride and causes slanted output if we trust
    // the raw pitch. When the delta is small, prefer the aligned geometry.
    if bpp > 0 && eff_width % 8 != 0 {
        let aligned_width = eff_width & !7;
        if aligned_width > 0 {
            let aligned_pitch = aligned_width.saturating_mul(bpp);
            let delta = if eff_pitch > aligned_pitch {
                eff_pitch - aligned_pitch
            } else {
                aligned_pitch - eff_pitch
            };
            if delta <= bpp.saturating_mul(8) && aligned_pitch % 64 == 0 {
                eff_width = aligned_width;
                eff_pitch = aligned_pitch;
            }
        }
    }

    (eff_width, eff_height, eff_pitch)
}

impl Framebuffer {
    pub fn new(fb: &limine::framebuffer::Framebuffer) -> Self {
        // Limine reports `bpp` in bits; clamp to supported 24/32-bit formats.
        let bits_per_pixel = fb.bpp() as u32;
        let pitch = fb.pitch() as u32;
        let width = fb.width() as u32;

        // Some firmware reports 24bpp but aligns pitch to 4 bytes per pixel.
        let pitch_bytes_per_pixel = if width > 0 { pitch / width } else { 0 };

        let bpp = match bits_per_pixel {
            16 => 2,
            24 => {
                if pitch_bytes_per_pixel >= 4 {
                    4
                } else {
                    3
                }
            }
            32 => 4,
            _ => {
                // Fallback: infer from pitch if sane, otherwise assume 4
                if pitch_bytes_per_pixel >= 4 {
                    4
                } else if pitch_bytes_per_pixel == 3 {
                    3
                } else {
                    4
                }
            }
        };

        let (width, height, pitch) = normalize_geometry(width, fb.height() as u32, pitch, bpp);

        Self {
            addr: fb.addr() as *mut u32,
            width,
            height,
            pitch,
            bpp,
        }
    }

    pub fn clear(&mut self, color: u32) {
        // Fast blit clear: fill first row, then copy to remaining rows
        let buf_len = (self.pitch as usize).saturating_mul(self.height as usize);
        let buffer = unsafe { core::slice::from_raw_parts_mut(self.addr as *mut u8, buf_len) };

        let bpp = self.bpp.max(1) as usize;
        let row_bytes = self.pitch as usize;

        if buf_len == 0 || row_bytes == 0 {
            return;
        }

        // For 32bpp, use u32 writes for even faster filling
        if bpp == 4 && row_bytes % 4 == 0 {
            // Cast to u32 slice for fast 4-byte writes
            let buffer_u32 = unsafe { core::slice::from_raw_parts_mut(self.addr, buf_len / 4) };
            let pixels_per_row = row_bytes / 4;

            // Fill first row with u32 writes
            for i in 0..pixels_per_row.min(self.width as usize) {
                buffer_u32[i] = color;
            }

            // Copy first row to all remaining rows using fast slice copy
            for y in 1..self.height as usize {
                let dst_start = y * pixels_per_row;
                if dst_start + pixels_per_row > buffer_u32.len() {
                    break;
                }
                buffer_u32.copy_within(0..pixels_per_row, dst_start);
            }
        } else {
            // Fallback for other bpp: fill first row byte-by-byte, then copy
            let color_bytes = color.to_le_bytes();
            let row_payload = (self.width as usize).saturating_mul(bpp).min(row_bytes);

            // Fill first row
            for x in 0..self.width as usize {
                let offset = x * bpp;
                if offset + bpp > row_bytes {
                    break;
                }
                for b in 0..bpp.min(4) {
                    buffer[offset + b] = color_bytes[b];
                }
            }

            // Copy first row to remaining rows
            for y in 1..self.height as usize {
                let dst_start = y * row_bytes;
                if dst_start + row_payload > buf_len {
                    break;
                }
                buffer.copy_within(0..row_payload, dst_start);
            }
        }
    }
}

pub fn get_info() -> Option<FramebufferInfo> {
    if let Some(resp) = FRAMEBUFFER_REQUEST.get_response() {
        if let Some(fb) = resp.framebuffers().into_iter().next() {
            let pitch = fb.pitch() as u32;
            let width = fb.width() as u32;
            let height = fb.height() as u32;
            let bits_per_pixel = fb.bpp() as u32;
            let pitch_bytes_per_pixel = if width > 0 { pitch / width } else { 0 };
            let bpp = match bits_per_pixel {
                16 => 2,
                24 => {
                    if pitch_bytes_per_pixel >= 4 {
                        4
                    } else {
                        3
                    }
                }
                32 => 4,
                _ => {
                    if pitch_bytes_per_pixel >= 4 {
                        4
                    } else if pitch_bytes_per_pixel == 3 {
                        3
                    } else {
                        4
                    }
                }
            };
            let (width, height, pitch) = normalize_geometry(width, height, pitch, bpp);

            return Some(FramebufferInfo {
                addr: fb.addr() as u64,
                // Keep the raw byte_len for full mapping safety.
                byte_len: fb.pitch() * fb.height(),
                width,
                height,
                pitch,
                bpp: fb.bpp() as u16,
                format: match fb.memory_model() {
                    // Limine "RGB" memory model is actually BGRX in memory layout
                    limine::framebuffer::MemoryModel::RGB => PixelFormat::Bgrx8888,
                    _ => PixelFormat::Unknown,
                },
            });
        }
    }
    None
}
