//! Font IPC Protocol - Control-plane messages for fontd service.
//!
//! This module defines the atlas-based, batched IPC protocol for font requests.
//! Glyph pixels are transferred via Bytespace, not inline in IPC messages.

extern crate alloc;

use crate::ids::HandleId;
use crate::wire::ThingId;
use alloc::vec::Vec;

/// Atlas pixel format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AtlasFormat {
    /// Grayscale alpha mask (1 byte per pixel) - preferred
    A8 = 0,
    /// Full RGBA (4 bytes per pixel) - fallback
    Rgba8888 = 1,
}

impl From<u8> for AtlasFormat {
    fn from(v: u8) -> Self {
        match v {
            0 => AtlasFormat::A8,
            1 => AtlasFormat::Rgba8888,
            _ => AtlasFormat::A8,
        }
    }
}

/// Request face metrics for layout
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct GetFaceMetrics {
    pub face_id: ThingId,
    pub px_size: u16,
}

/// Face metrics response
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct FaceMetrics {
    pub ascent: i16,
    pub descent: i16,
    pub line_gap: i16,
    pub units_per_em: u16,
}

/// Batch request for glyphs - the core IPC primitive
#[derive(Debug, Clone)]
pub struct EnsureGlyphs {
    pub face_id: ThingId,
    pub px_size: u16,
    pub glyph_ids: Vec<u32>, // Codepoints
}

/// Single glyph placement in atlas
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct GlyphPlacement {
    pub glyph_id: u32,
    pub x: u16, // Atlas rect position
    pub y: u16,
    pub w: u16, // Atlas rect size
    pub h: u16,
    pub bearing_x: i16, // X offset from origin
    pub bearing_y: i16, // Y offset from baseline
    pub advance: i16,   // Horizontal advance
}

/// Response to EnsureGlyphs - placements + atlas info
#[derive(Debug, Clone)]
pub struct EnsureGlyphsResp {
    pub req_face_id: ThingId,
    pub req_px_size: u16,
    pub atlas_fd: u32,
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub atlas_format: AtlasFormat,
    pub atlas_version: u64, // Monotonic per (face, size)
    pub placements: Vec<GlyphPlacement>,
    pub missing: Vec<u32>, // Glyphs that couldn't be produced
}

/// Font protocol errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FontError {
    UnknownFace = 1,
    BadSize = 2,
    RasterFailed = 3,
    AtlasAllocationFailed = 4,
    EncodingError = 5,
}

/// IPC message tag for font requests
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FontRequestTag {
    Ping = 0,
    GetFaceMetrics = 1,
    EnsureGlyphs = 2,
}

/// IPC message tag for font responses
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FontResponseTag {
    Pong = 0,
    FaceMetrics = 1,
    EnsureGlyphsResp = 2,
    Error = 255,
}

// ============================================================================
// Wire encoding/decoding
// ============================================================================

impl GetFaceMetrics {
    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < 1 + 8 + 2 {
            return None;
        }
        buf[0] = FontRequestTag::GetFaceMetrics as u8;
        buf[1..9].copy_from_slice(&self.face_id.to_u64_lossy().to_le_bytes());
        buf[9..11].copy_from_slice(&self.px_size.to_le_bytes());
        Some(11)
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 10 {
            return None;
        }
        let face_id = ThingId::from_u64(u64::from_le_bytes(buf[0..8].try_into().ok()?));
        let px_size = u16::from_le_bytes(buf[8..10].try_into().ok()?);
        Some(Self { face_id, px_size })
    }
}

impl FaceMetrics {
    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < 1 + 8 {
            return None;
        }
        buf[0] = FontResponseTag::FaceMetrics as u8;
        buf[1..3].copy_from_slice(&self.ascent.to_le_bytes());
        buf[3..5].copy_from_slice(&self.descent.to_le_bytes());
        buf[5..7].copy_from_slice(&self.line_gap.to_le_bytes());
        buf[7..9].copy_from_slice(&self.units_per_em.to_le_bytes());
        Some(9)
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 8 {
            return None;
        }
        Some(Self {
            ascent: i16::from_le_bytes(buf[0..2].try_into().ok()?),
            descent: i16::from_le_bytes(buf[2..4].try_into().ok()?),
            line_gap: i16::from_le_bytes(buf[4..6].try_into().ok()?),
            units_per_em: u16::from_le_bytes(buf[6..8].try_into().ok()?),
        })
    }
}

impl EnsureGlyphs {
    /// Encode EnsureGlyphs request
    /// Format: tag(1) + face_id(8) + px_size(2) + count(4) + glyph_ids(count*4)
    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        let count = self.glyph_ids.len();
        let needed = 1 + 8 + 2 + 4 + count * 4;
        if buf.len() < needed {
            return None;
        }

        buf[0] = FontRequestTag::EnsureGlyphs as u8;
        buf[1..9].copy_from_slice(&self.face_id.to_u64_lossy().to_le_bytes());
        buf[9..11].copy_from_slice(&self.px_size.to_le_bytes());
        buf[11..15].copy_from_slice(&(count as u32).to_le_bytes());

        let mut offset = 15;
        for gid in &self.glyph_ids {
            buf[offset..offset + 4].copy_from_slice(&gid.to_le_bytes());
            offset += 4;
        }
        Some(offset)
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 14 {
            return None;
        }
        let face_id = ThingId::from_u64(u64::from_le_bytes(buf[0..8].try_into().ok()?));
        let px_size = u16::from_le_bytes(buf[8..10].try_into().ok()?);
        let count = u32::from_le_bytes(buf[10..14].try_into().ok()?) as usize;

        if buf.len() < 14 + count * 4 {
            return None;
        }
        let mut glyph_ids = Vec::with_capacity(count);
        let mut offset = 14;
        for _ in 0..count {
            glyph_ids.push(u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?));
            offset += 4;
        }
        Some(Self {
            face_id,
            px_size,
            glyph_ids,
        })
    }
}

impl GlyphPlacement {
    pub const WIRE_SIZE: usize = 4 + 2 * 4 + 2 * 3; // glyph_id + x,y,w,h + bearing_x,bearing_y,advance

    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < Self::WIRE_SIZE {
            return None;
        }
        buf[0..4].copy_from_slice(&self.glyph_id.to_le_bytes());
        buf[4..6].copy_from_slice(&self.x.to_le_bytes());
        buf[6..8].copy_from_slice(&self.y.to_le_bytes());
        buf[8..10].copy_from_slice(&self.w.to_le_bytes());
        buf[10..12].copy_from_slice(&self.h.to_le_bytes());
        buf[12..14].copy_from_slice(&self.bearing_x.to_le_bytes());
        buf[14..16].copy_from_slice(&self.bearing_y.to_le_bytes());
        buf[16..18].copy_from_slice(&self.advance.to_le_bytes());
        Some(18)
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::WIRE_SIZE {
            return None;
        }
        Some(Self {
            glyph_id: u32::from_le_bytes(buf[0..4].try_into().ok()?),
            x: u16::from_le_bytes(buf[4..6].try_into().ok()?),
            y: u16::from_le_bytes(buf[6..8].try_into().ok()?),
            w: u16::from_le_bytes(buf[8..10].try_into().ok()?),
            h: u16::from_le_bytes(buf[10..12].try_into().ok()?),
            bearing_x: i16::from_le_bytes(buf[12..14].try_into().ok()?),
            bearing_y: i16::from_le_bytes(buf[14..16].try_into().ok()?),
            advance: i16::from_le_bytes(buf[16..18].try_into().ok()?),
        })
    }
}

impl EnsureGlyphsResp {
    /// Encode EnsureGlyphsResp
    /// Format: tag(1) + req_face(8) + req_px(2) + atlas_fd(4) + w(4) + h(4) + fmt(1) + ver(8) + ...
    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        let place_count = self.placements.len();
        let miss_count = self.missing.len();
        let needed = 1 + 8 + 2 + 4 + 4 + 4 + 1 + 8 + 4 + place_count * 18 + 4 + miss_count * 4;
        if buf.len() < needed {
            return None;
        }

        buf[0] = FontResponseTag::EnsureGlyphsResp as u8;
        buf[1..9].copy_from_slice(&self.req_face_id.to_u64_lossy().to_le_bytes());
        buf[9..11].copy_from_slice(&self.req_px_size.to_le_bytes());
        buf[11..15].copy_from_slice(&self.atlas_fd.to_le_bytes());
        buf[15..19].copy_from_slice(&self.atlas_width.to_le_bytes());
        buf[19..23].copy_from_slice(&self.atlas_height.to_le_bytes());
        buf[23] = self.atlas_format as u8;
        buf[24..32].copy_from_slice(&self.atlas_version.to_le_bytes());
        buf[32..36].copy_from_slice(&(place_count as u32).to_le_bytes());

        let mut offset = 36;
        for p in &self.placements {
            p.encode(&mut buf[offset..])?;
            offset += 18;
        }

        buf[offset..offset + 4].copy_from_slice(&(miss_count as u32).to_le_bytes());
        offset += 4;
        for m in &self.missing {
            buf[offset..offset + 4].copy_from_slice(&m.to_le_bytes());
            offset += 4;
        }

        Some(offset)
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 36 {
            return None;
        }
        let req_face_id = ThingId::from_u64(u64::from_le_bytes(buf[0..8].try_into().ok()?));
        let req_px_size = u16::from_le_bytes(buf[8..10].try_into().ok()?);
        let atlas_fd = u32::from_le_bytes(buf[10..14].try_into().ok()?);
        let atlas_width = u32::from_le_bytes(buf[14..18].try_into().ok()?);
        let atlas_height = u32::from_le_bytes(buf[18..22].try_into().ok()?);
        let atlas_format = AtlasFormat::from(buf[22]);
        let atlas_version = u64::from_le_bytes(buf[23..31].try_into().ok()?);
        let place_count = u32::from_le_bytes(buf[31..35].try_into().ok()?) as usize;

        let mut offset = 35;
        let mut placements = Vec::with_capacity(place_count);
        for _ in 0..place_count {
            if offset + 18 > buf.len() {
                return None;
            }
            placements.push(GlyphPlacement::decode(&buf[offset..])?);
            offset += 18;
        }

        if offset + 4 > buf.len() {
            return None;
        }
        let miss_count = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
        offset += 4;

        let mut missing = Vec::with_capacity(miss_count);
        for _ in 0..miss_count {
            if offset + 4 > buf.len() {
                return None;
            }
            missing.push(u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?));
            offset += 4;
        }

        Some(Self {
            req_face_id,
            req_px_size,
            atlas_fd,
            atlas_width,
            atlas_height,
            atlas_format,
            atlas_version,
            placements,
            missing,
        })
    }
}

/// Encode a Ping request
pub fn encode_ping(buf: &mut [u8]) -> Option<usize> {
    if buf.is_empty() {
        return None;
    }
    buf[0] = FontRequestTag::Ping as u8;
    Some(1)
}

/// Encode a Pong response
pub fn encode_pong(buf: &mut [u8]) -> Option<usize> {
    if buf.is_empty() {
        return None;
    }
    buf[0] = FontResponseTag::Pong as u8;
    Some(1)
}

/// Encode an error response
pub fn encode_error(err: FontError, buf: &mut [u8]) -> Option<usize> {
    if buf.len() < 2 {
        return None;
    }
    buf[0] = FontResponseTag::Error as u8;
    buf[1] = err as u8;
    Some(2)
}

/// Decode request tag from buffer
pub fn decode_request_tag(buf: &[u8]) -> Option<FontRequestTag> {
    if buf.is_empty() {
        return None;
    }
    match buf[0] {
        0 => Some(FontRequestTag::Ping),
        1 => Some(FontRequestTag::GetFaceMetrics),
        2 => Some(FontRequestTag::EnsureGlyphs),
        _ => None,
    }
}

/// Decode response tag from buffer
pub fn decode_response_tag(buf: &[u8]) -> Option<FontResponseTag> {
    if buf.is_empty() {
        return None;
    }
    match buf[0] {
        0 => Some(FontResponseTag::Pong),
        1 => Some(FontResponseTag::FaceMetrics),
        2 => Some(FontResponseTag::EnsureGlyphsResp),
        255 => Some(FontResponseTag::Error),
        _ => None,
    }
}
