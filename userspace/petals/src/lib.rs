#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;

pub mod blit;
pub mod bmp;
pub mod font;
pub mod geometry;
pub mod raster;
pub mod tessellate;

use abi::pixel::PixelFormat;
use abi::vm::{VmBacking, VmMapFlags, VmMapReq, VmProt};
use alloc::vec::Vec;
use stem::syscall::{memfd_create, vm_map, vm_unmap};

pub struct Texture {
    pub fd: u32,
    pub ptr: *mut u8,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub bpp: u8,
    pub size: usize,
}

impl Texture {
    pub fn new(name: &str, width: u32, height: u32, bpp: u8) -> Option<Self> {
        let stride = width * bpp as u32;
        let size = (height as usize) * (stride as usize);
        let fd = memfd_create(name, size).ok()? as u32;

        let req = VmMapReq {
            addr_hint: 0,
            len: size,
            prot: VmProt::READ | VmProt::WRITE | VmProt::USER,
            flags: VmMapFlags::SHARED,
            backing: VmBacking::File { fd, offset: 0 },
        };

        let resp = vm_map(&req).ok()?;

        Some(Self {
            fd,
            ptr: resp.addr as *mut u8,
            width,
            height,
            stride,
            bpp,
            size,
        })
    }

    pub fn as_slice_mut(&mut self) -> &mut [u32] {
        if self.bpp != 4 {
            panic!("Texture is not 4bpp");
        }
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut u32, self.size / 4) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        let _ = vm_unmap(self.ptr as usize, self.size);
        // Note: FD leaking might be an issue if we don't close it,
        // but petals doesn't own the lifecycle of the FD in all cases.
    }
}

pub struct Canvas<'a> {
    pub buffer: &'a mut [u32],
    pub width: u32,
    pub height: u32,
    pub stride_pixels: u32,
}

impl<'a> Canvas<'a> {
    pub fn new(buffer: &'a mut [u32], width: u32, height: u32, stride_pixels: u32) -> Self {
        Self {
            buffer,
            width,
            height,
            stride_pixels,
        }
    }

    pub fn from_texture(texture: &'a mut Texture) -> Self {
        let width = texture.width;
        let height = texture.height;
        let stride_pixels = texture.stride / 4;
        Self {
            buffer: texture.as_slice_mut(),
            width,
            height,
            stride_pixels,
        }
    }

    pub fn clear(&mut self, color: u32) {
        self.buffer.fill(color);
    }

    pub fn put_pixel(&mut self, x: u32, y: u32, color: u32) {
        if x < self.width && y < self.height {
            self.buffer[(y * self.stride_pixels + x) as usize] = color;
        }
    }

    pub fn blit(
        &mut self,
        src: &[u32],
        src_width: u32,
        src_height: u32,
        src_stride_pixels: u32,
        dst_x: i32,
        dst_y: i32,
    ) {
        for sy in 0..src_height {
            let dy = dst_y + sy as i32;
            if dy < 0 || dy >= self.height as i32 {
                continue;
            }

            for sx in 0..src_width {
                let dx = dst_x + sx as i32;
                if dx < 0 || dx >= self.width as i32 {
                    continue;
                }

                let color = src[(sy * src_stride_pixels + sx) as usize];
                // Simple alpha blending if needed, but for wallpaper we just blit.
                if (color >> 24) == 0xFF {
                    self.buffer[(dy as u32 * self.stride_pixels + dx as u32) as usize] = color;
                } else if (color >> 24) > 0 {
                    // Basic alpha blending
                    let dst_idx = (dy as u32 * self.stride_pixels + dx as u32) as usize;
                    let dst_color = self.buffer[dst_idx];
                    self.buffer[dst_idx] = blend(dst_color, color);
                }
            }
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

// ── Atlas Packer ──────────────────────────────────────────────────────────────

pub struct Atlas {
    pub texture: Texture,
    pub next_x: u32,
    pub next_y: u32,
    pub row_h: u32,
    pub padding: u32,
}

impl Atlas {
    pub fn new(name: &str, w: u32, h: u32, bpp: u8) -> Option<Self> {
        let mut texture = Texture::new(name, w, h, bpp)?;
        texture.as_bytes_mut().fill(0);
        Some(Self {
            texture,
            next_x: 0,
            next_y: 0,
            row_h: 0,
            padding: 1,
        })
    }

    pub fn pack(&mut self, w: u32, h: u32, pixels: &[u8]) -> Option<(u32, u32)> {
        if self.next_x + w + self.padding > self.texture.width {
            self.next_x = 0;
            self.next_y += self.row_h + self.padding;
            self.row_h = 0;
        }

        if self.next_y + h + self.padding > self.texture.height {
            return None; // Full
        }

        let x = self.next_x;
        let y = self.next_y;

        // Blit to atlas
        let atlas_stride = self.texture.stride as usize;
        let tex_buf = self.texture.as_bytes_mut();
        for sy in 0..h {
            let dy = y + sy;
            let dst_off = dy as usize * atlas_stride + x as usize;
            let src_off = sy as usize * w as usize;
            tex_buf[dst_off..dst_off + w as usize]
                .copy_from_slice(&pixels[src_off..src_off + w as usize]);
        }

        self.next_x += w + self.padding;
        self.row_h = self.row_h.max(h);

        Some((x, y))
    }
}
