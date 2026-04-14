//! Wire-safe geometry primitives for graph payloads.
//!
//! All numeric fields are serialized as little-endian bytes.
//! Use `as_bytes`/`from_bytes` for wire conversion; avoid transmuting raw bytes.

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RectI32Wire {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl RectI32Wire {
    pub const fn new(x: i32, y: i32, w: i32, h: i32) -> Self {
        Self { x, y, w, h }
    }

    pub const fn as_bytes(&self) -> [u8; 16] {
        let x = self.x.to_le_bytes();
        let y = self.y.to_le_bytes();
        let w = self.w.to_le_bytes();
        let h = self.h.to_le_bytes();
        [
            x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], w[0], w[1], w[2], w[3], h[0], h[1],
            h[2], h[3],
        ]
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        let x = i32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let y = i32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let w = i32::from_le_bytes(bytes[8..12].try_into().ok()?);
        let h = i32::from_le_bytes(bytes[12..16].try_into().ok()?);
        Some(Self { x, y, w, h })
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Mat3x2fWire {
    pub m11: f32,
    pub m12: f32,
    pub m21: f32,
    pub m22: f32,
    pub dx: f32,
    pub dy: f32,
}

impl Mat3x2fWire {
    pub const fn identity() -> Self {
        Self {
            m11: 1.0,
            m12: 0.0,
            m21: 0.0,
            m22: 1.0,
            dx: 0.0,
            dy: 0.0,
        }
    }

    pub fn as_bytes(&self) -> [u8; 24] {
        let m11 = self.m11.to_le_bytes();
        let m12 = self.m12.to_le_bytes();
        let m21 = self.m21.to_le_bytes();
        let m22 = self.m22.to_le_bytes();
        let dx = self.dx.to_le_bytes();
        let dy = self.dy.to_le_bytes();
        [
            m11[0], m11[1], m11[2], m11[3], m12[0], m12[1], m12[2], m12[3], m21[0], m21[1], m21[2],
            m21[3], m22[0], m22[1], m22[2], m22[3], dx[0], dx[1], dx[2], dx[3], dy[0], dy[1],
            dy[2], dy[3],
        ]
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 24 {
            return None;
        }
        let m11 = f32::from_le_bytes(bytes[0..4].try_into().ok()?);
        let m12 = f32::from_le_bytes(bytes[4..8].try_into().ok()?);
        let m21 = f32::from_le_bytes(bytes[8..12].try_into().ok()?);
        let m22 = f32::from_le_bytes(bytes[12..16].try_into().ok()?);
        let dx = f32::from_le_bytes(bytes[16..20].try_into().ok()?);
        let dy = f32::from_le_bytes(bytes[20..24].try_into().ok()?);
        Some(Self {
            m11,
            m12,
            m21,
            m22,
            dx,
            dy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_i32_roundtrip() {
        let rect = RectI32Wire::new(-2, 4, 16, 32);
        let bytes = rect.as_bytes();
        let decoded = RectI32Wire::from_bytes(&bytes).expect("decode");
        assert_eq!(rect, decoded);
    }

    #[test]
    fn mat3x2_roundtrip() {
        let mat = Mat3x2fWire {
            m11: 1.0,
            m12: 2.0,
            m21: 3.0,
            m22: 4.0,
            dx: 5.0,
            dy: 6.0,
        };
        let bytes = mat.as_bytes();
        let decoded = Mat3x2fWire::from_bytes(&bytes).expect("decode");
        assert_eq!(mat, decoded);
    }
}
