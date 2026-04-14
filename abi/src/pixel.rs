//! Canonical Pixel Format Definitions for Thing-OS
//!
//! The canonical format is B8G8R8A8 in memory order, which when read
//! as a little-endian u32 gives 0xAARRGGBB.

/// Pixel format enumeration with explicit byte ordering.
///
/// All formats describe byte order in memory (LSB first on LE systems).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PixelFormat {
    /// Unknown or unsupported format.
    Unknown = 0,

    /// 32-bit BGRA: Memory order is [B, G, R, A].
    /// As little-endian u32: 0xAARRGGBB.
    /// Alpha 255 = opaque, 0 = transparent.
    Bgra8888 = 1,

    /// 32-bit BGRX: Memory order is [B, G, R, X].
    /// Alpha byte ignored (treated as opaque).
    /// As little-endian u32: 0xXXRRGGBB.
    Bgrx8888 = 2,

    /// 16-bit RGB565: Memory order is [low, high].
    /// No alpha channel.
    Rgb565 = 3,
}

impl PixelFormat {
    /// Convert to wire-compatible u64 value for graph properties.
    pub const fn to_wire(self) -> u64 {
        self as u64
    }

    /// Parse from wire u64 value.
    pub const fn from_wire(v: u64) -> Self {
        match v {
            1 => Self::Bgra8888,
            2 => Self::Bgrx8888,
            3 => Self::Rgb565,
            _ => Self::Unknown,
        }
    }

    /// Bytes per pixel for this format.
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Bgra8888 | Self::Bgrx8888 => 4,
            Self::Rgb565 => 2,
            Self::Unknown => 0,
        }
    }

    /// Whether this format has an alpha channel.
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::Bgra8888)
    }
}

/// Canonical color representation.
///
/// Stored internally as 0xAARRGGBB (little-endian BGRA memory layout).
/// When written to memory via `to_le_bytes()`, the byte order is [B, G, R, A].
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Color(pub u32);

impl Color {
    /// Pure black, fully opaque.
    pub const BLACK: Self = Self::from_rgb(0, 0, 0);
    /// Pure white, fully opaque.
    pub const WHITE: Self = Self::from_rgb(255, 255, 255);
    /// Pure red, fully opaque.
    pub const RED: Self = Self::from_rgb(255, 0, 0);
    /// Pure green, fully opaque.
    pub const GREEN: Self = Self::from_rgb(0, 255, 0);
    /// Pure blue, fully opaque.
    pub const BLUE: Self = Self::from_rgb(0, 0, 255);
    /// Fully transparent.
    pub const TRANSPARENT: Self = Self::from_rgba(0, 0, 0, 0);

    /// Create color from RGBA components.
    #[inline]
    pub const fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self(((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32))
    }

    /// Create fully opaque color from RGB components.
    #[inline]
    pub const fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self::from_rgba(r, g, b, 255)
    }

    /// Create color from raw 0xAARRGGBB value.
    #[inline]
    pub const fn from_argb_u32(v: u32) -> Self {
        Self(v)
    }

    /// Get alpha component (0 = transparent, 255 = opaque).
    #[inline]
    pub const fn alpha(self) -> u8 {
        (self.0 >> 24) as u8
    }

    /// Get red component.
    #[inline]
    pub const fn red(self) -> u8 {
        (self.0 >> 16) as u8
    }

    /// Get green component.
    #[inline]
    pub const fn green(self) -> u8 {
        (self.0 >> 8) as u8
    }

    /// Get blue component.
    #[inline]
    pub const fn blue(self) -> u8 {
        self.0 as u8
    }

    /// Get raw 0xAARRGGBB value.
    #[inline]
    pub const fn to_argb_u32(self) -> u32 {
        self.0
    }

    /// Convert to little-endian bytes for writing to BGRA framebuffer.
    ///
    /// Returns [B, G, R, A] in memory order.
    #[inline]
    pub const fn to_le_bytes(self) -> [u8; 4] {
        self.0.to_le_bytes()
    }

    /// Create color from little-endian bytes read from BGRA framebuffer.
    ///
    /// Expects [B, G, R, A] in memory order.
    #[inline]
    pub const fn from_le_bytes(bytes: [u8; 4]) -> Self {
        Self(u32::from_le_bytes(bytes))
    }

    /// Apply alpha pre-multiplication.
    #[inline]
    pub const fn premultiply(self) -> Self {
        let a = self.alpha() as u32;
        if a == 255 {
            return self;
        }
        if a == 0 {
            return Self::TRANSPARENT;
        }
        let r = ((self.red() as u32) * a / 255) as u8;
        let g = ((self.green() as u32) * a / 255) as u8;
        let b = ((self.blue() as u32) * a / 255) as u8;
        Self::from_rgba(r, g, b, self.alpha())
    }

    /// Set alpha component, preserving RGB.
    #[inline]
    pub const fn with_alpha(self, a: u8) -> Self {
        Self((self.0 & 0x00FFFFFF) | ((a as u32) << 24))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_byte_order() {
        // Red: 0xFFFF0000 -> [B=0, G=0, R=255, A=255]
        let red = Color::from_rgb(255, 0, 0);
        assert_eq!(red.to_le_bytes(), [0, 0, 255, 255]);

        // Green: 0xFF00FF00 -> [B=0, G=255, R=0, A=255]
        let green = Color::from_rgb(0, 255, 0);
        assert_eq!(green.to_le_bytes(), [0, 255, 0, 255]);

        // Blue: 0xFF0000FF -> [B=255, G=0, R=0, A=255]
        let blue = Color::from_rgb(0, 0, 255);
        assert_eq!(blue.to_le_bytes(), [255, 0, 0, 255]);
    }

    #[test]
    fn color_channel_extraction() {
        let c = Color::from_rgba(0xAA, 0xBB, 0xCC, 0xDD);
        assert_eq!(c.red(), 0xAA);
        assert_eq!(c.green(), 0xBB);
        assert_eq!(c.blue(), 0xCC);
        assert_eq!(c.alpha(), 0xDD);
    }

    #[test]
    fn color_roundtrip() {
        let original = Color::from_rgba(100, 150, 200, 128);
        let bytes = original.to_le_bytes();
        let restored = Color::from_le_bytes(bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn pixel_format_wire_roundtrip() {
        for fmt in [
            PixelFormat::Unknown,
            PixelFormat::Bgra8888,
            PixelFormat::Bgrx8888,
            PixelFormat::Rgb565,
        ] {
            assert_eq!(PixelFormat::from_wire(fmt.to_wire()), fmt);
        }
    }

    #[test]
    fn test_pixel_format_properties() {
        assert_eq!(PixelFormat::Bgra8888.bytes_per_pixel(), 4);
        assert!(PixelFormat::Bgra8888.has_alpha());

        assert_eq!(PixelFormat::Bgrx8888.bytes_per_pixel(), 4);
        assert!(!PixelFormat::Bgrx8888.has_alpha());

        assert_eq!(PixelFormat::Rgb565.bytes_per_pixel(), 2);
        assert!(!PixelFormat::Rgb565.has_alpha());

        assert_eq!(PixelFormat::Unknown.bytes_per_pixel(), 0);
        assert!(!PixelFormat::Unknown.has_alpha());
    }

    #[test]
    fn test_premultiply() {
        // Opaque - no change
        let opaque = Color::from_rgba(100, 150, 200, 255);
        assert_eq!(opaque.premultiply(), opaque);

        // Transparent - becomes transparent black
        let transparent = Color::from_rgba(100, 150, 200, 0);
        assert_eq!(transparent.premultiply(), Color::TRANSPARENT);

        // Semi-transparent
        let semi = Color::from_rgba(255, 128, 64, 128); // approx 50% opacity
        let premul = semi.premultiply();

        // 255 * 128 / 255 = 128
        assert_eq!(premul.red(), 128);
        // 128 * 128 / 255 = 64
        assert_eq!(premul.green(), 64);
        // 64 * 128 / 255 = 32
        assert_eq!(premul.blue(), 32);
        // Alpha remains 128
        assert_eq!(premul.alpha(), 128);
    }
}
