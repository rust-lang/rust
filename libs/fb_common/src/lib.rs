#![no_std]

/// Simple pixel format list shared between boot-time drawing code and
/// framebuffer helpers. Byte order is little-endian in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgb888,   // [R, G, B]
    Bgr888,   // [B, G, R]
    Bgrx8888, // [B, G, R, X]
    Rgbx8888, // [R, G, B, X]
    Rgb565,   // 16-bit 5:6:5, little-endian
    Unknown,
}

impl PixelFormat {
    #[inline]
    pub const fn bytes_per_pixel(self) -> u32 {
        match self {
            PixelFormat::Bgrx8888 | PixelFormat::Rgbx8888 => 4,
            PixelFormat::Rgb888 | PixelFormat::Bgr888 => 3,
            PixelFormat::Rgb565 => 2,
            PixelFormat::Unknown => 0,
        }
    }
}

/// Normalize a reported stride (pitch).
///
/// - `reported_stride` is assumed to be bytes if >= width * bpp.
/// - If smaller, treat it as pixels-per-row and scale by bpp.
/// - Fallback to tight packing if 0 or still smaller than the row payload.
#[inline]
pub const fn calc_stride_bytes(width: u32, bpp: u32, reported_stride: u32) -> u32 {
    let row_bytes = width.saturating_mul(bpp);

    if reported_stride == 0 {
        return row_bytes;
    }

    if reported_stride >= row_bytes {
        return reported_stride;
    }

    let scaled = reported_stride.saturating_mul(bpp);
    if scaled >= row_bytes {
        scaled
    } else {
        row_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    #[test]
    fn test_pixel_format_bpp() {
        assert_eq!(PixelFormat::Rgb888.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Bgr888.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Bgrx8888.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Rgbx8888.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Rgb565.bytes_per_pixel(), 2);
        assert_eq!(PixelFormat::Unknown.bytes_per_pixel(), 0);
    }

    #[test]
    fn test_calc_stride_bytes() {
        // Case 1: reported_stride is 0 -> tight packing
        // width=100, bpp=4 => row_bytes=400
        assert_eq!(calc_stride_bytes(100, 4, 0), 400);

        // Case 2: reported_stride >= row_bytes -> use reported
        // width=100, bpp=4 => row_bytes=400
        // reported 400 => 400
        assert_eq!(calc_stride_bytes(100, 4, 400), 400);
        // reported 512 => 512
        assert_eq!(calc_stride_bytes(100, 4, 512), 512);

        // Case 3: reported_stride < row_bytes, interpret as pixels
        // width=100, bpp=4 => row_bytes=400
        // reported 100 (pixels) => 100 * 4 = 400
        assert_eq!(calc_stride_bytes(100, 4, 100), 400);
        // reported 128 (pixels) => 128 * 4 = 512
        assert_eq!(calc_stride_bytes(100, 4, 128), 512);

        // Case 4: Ambiguous case / small value
        // width=100, bpp=4 => row_bytes=400
        // reported 50. 50 < 400. 50 * 4 = 200. 200 < 400.
        // Should fallback to row_bytes (400)
        assert_eq!(calc_stride_bytes(100, 4, 50), 400);
    }
}
