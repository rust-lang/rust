#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use crate::geometry::Rect;
use stem::simd;

/// A raw pointer-based pixel buffer with dimensions and stride.
#[derive(Clone, Copy)]
pub struct PixelBuffer {
    pub ptr: *mut u32,
    pub width: u32,
    pub height: u32,
    pub stride_bytes: u32,
}

impl PixelBuffer {
    pub unsafe fn new(ptr: *mut u32, width: u32, height: u32, stride_bytes: u32) -> Self {
        Self {
            ptr,
            width,
            height,
            stride_bytes,
        }
    }

    pub fn as_slice_mut(&mut self) -> &mut [u32] {
        let len = (self.height as usize) * (self.stride_bytes as usize / 4);
        unsafe { core::slice::from_raw_parts_mut(self.ptr, len) }
    }

    pub fn row_mut(&mut self, y: u32) -> &mut [u32] {
        let offset = (y * self.stride_bytes / 4) as usize;
        let len = self.width as usize;
        unsafe { core::slice::from_raw_parts_mut(self.ptr.add(offset), len) }
    }
}

pub fn blit_rect(
    dst: &mut PixelBuffer,
    src: &PixelBuffer,
    src_rect: &Rect,
    dst_x: i32,
    dst_y: i32,
    clip: &Rect,
) {
    let intersect = clip.intersect(Rect::new(dst_x, dst_y, src_rect.width(), src_rect.height()));
    if intersect.is_empty() {
        return;
    }

    let dx_start = intersect.x();
    let dy_start = intersect.y();
    let width = intersect.width();
    let height = intersect.height();

    let sx_start = src_rect.x() + (dx_start - dst_x);
    let sy_start = src_rect.y() + (dy_start - dst_y);

    for y in 0..height {
        let d_row = dst.row_mut((dy_start + y) as u32);
        let s_row = unsafe {
            let offset = ((sy_start + y) as u32 * src.stride_bytes / 4) as usize;
            core::slice::from_raw_parts(src.ptr.add(offset), src.width as usize)
        };

        let d_span = &mut d_row[dx_start as usize..(dx_start + width) as usize];
        let s_span = &s_row[sx_start as usize..(sx_start + width) as usize];

        simd::blit_rgba8888_over(d_span, s_span);
    }
}
