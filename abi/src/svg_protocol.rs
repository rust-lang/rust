//! SVG IPC Protocol - Control-plane messages for blossom service.
//!
//! This module defines the wire-compatible IPC protocol for SVG rasterization requests.
//! Raster pixels are transferred via Bytespace, not inline in IPC messages.

extern crate alloc;

use crate::ids::HandleId;
use crate::wire::ThingId;
use alloc::vec::Vec;

/// Source of SVG data for rasterization
#[derive(Debug, Clone)]
pub enum SvgSource {
    /// SVG bytes are in a memfd
    MemFd(u32),
    /// SVG bytes are inline (for tests/debug only)
    InlineBytes(Vec<u8>),
}

/// Pixel format for rasterized output
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PixelFormat {
    /// BGRA8888 - matches Bloom's canonical format
    Bgra8888 = 1,
}

impl From<u8> for PixelFormat {
    fn from(v: u8) -> Self {
        match v {
            1 => PixelFormat::Bgra8888,
            _ => PixelFormat::Bgra8888, // Default
        }
    }
}

/// SVG rasterization request
#[derive(Debug, Clone)]
pub struct RasterizeSvgRequest {
    /// Source of SVG bytes
    pub source: SvgSource,
    /// Requested width in pixels
    pub width: u32,
    /// Requested height in pixels  
    pub height: u32,
    /// Output pixel format
    pub pixel_format: u8,
    /// Reserved flags (0 for now)
    pub flags: u32,
}

/// SVG rasterization response
#[derive(Debug, Clone)]
pub struct RasterizeSvgResponse {
    /// Status code
    pub status: u8,
    /// MemFD containing rasterized pixels
    pub raster_fd: u32,
    /// Actual width of rasterized image
    pub width: u32,
    /// Actual height of rasterized image
    pub height: u32,
    /// Stride in bytes per row
    pub stride_bytes: u32,
    /// Pixel format of output
    pub pixel_format: u8,
    /// Cache key for this variant
    pub variant_hash: u64,
}

/// SVG protocol status codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SvgStatus {
    Ok = 0,
    ErrInvalidSvg = 1,
    ErrOom = 2,
    ErrUnsupported = 3,
}

/// SVG protocol request tags
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SvgRequestTag {
    Ping = 0,
    RasterizeSvg = 1,
}

// ============================================================================
// Wire encoding/decoding
// ============================================================================

impl RasterizeSvgRequest {
    /// Encode request to wire format
    /// Format: tag(1) + source_type(1) + [source_data] + width(4) + height(4) + format(1) + flags(4)
    /// For MemFd: source_data = fd(4) + pad(4)
    /// For InlineBytes: source_data = len(4) + bytes(len)
    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        let mut pos = 0;

        // Tag
        if pos >= buf.len() {
            return None;
        }
        buf[pos] = SvgRequestTag::RasterizeSvg as u8;
        pos += 1;

        // Source
        match &self.source {
            SvgSource::MemFd(fd) => {
                if pos >= buf.len() {
                    return None;
                }
                buf[pos] = 0; // MemFd type
                pos += 1;

                if pos + 8 > buf.len() {
                    return None;
                }
                buf[pos..pos + 4].copy_from_slice(&fd.to_le_bytes());
                buf[pos + 4..pos + 8].fill(0);
                pos += 8;
            }
            SvgSource::InlineBytes(bytes) => {
                if pos >= buf.len() {
                    return None;
                }
                buf[pos] = 1; // Inline type
                pos += 1;

                if pos + 4 > buf.len() {
                    return None;
                }
                buf[pos..pos + 4].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
                pos += 4;

                if pos + bytes.len() > buf.len() {
                    return None;
                }
                buf[pos..pos + bytes.len()].copy_from_slice(bytes);
                pos += bytes.len();
            }
        }

        // Width, Height, Format, Flags
        if pos + 13 > buf.len() {
            return None;
        }
        buf[pos..pos + 4].copy_from_slice(&self.width.to_le_bytes());
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.height.to_le_bytes());
        pos += 4;
        buf[pos] = self.pixel_format;
        pos += 1;
        buf[pos..pos + 4].copy_from_slice(&self.flags.to_le_bytes());
        pos += 4;

        Some(pos)
    }

    /// Decode request from wire format (excluding tag byte)
    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.is_empty() {
            return None;
        }

        let mut pos = 0;

        // Source type
        let source_type = buf[pos];
        pos += 1;

        let source = match source_type {
            0 => {
                // MemFd
                if pos + 8 > buf.len() {
                    return None;
                }
                let fd = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
                pos += 8;
                SvgSource::MemFd(fd)
            }
            1 => {
                // Inline
                if pos + 4 > buf.len() {
                    return None;
                }
                let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
                pos += 4;

                if pos + len > buf.len() {
                    return None;
                }
                let bytes = buf[pos..pos + len].to_vec();
                pos += len;
                SvgSource::InlineBytes(bytes)
            }
            _ => return None,
        };

        // Width, Height, Format, Flags
        if pos + 13 > buf.len() {
            return None;
        }
        let width = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let height = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let pixel_format = buf[pos];
        pos += 1;
        let flags = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);

        Some(Self {
            source,
            width,
            height,
            pixel_format,
            flags,
        })
    }
}

impl RasterizeSvgResponse {
    /// Encode response to wire format
    /// Format: status(1) + raster_fd(4) + pad(4) + width(4) + height(4) + stride(4) + format(1) + variant_hash(8)
    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        const SIZE: usize = 1 + 8 + 4 + 4 + 4 + 1 + 8;
        if buf.len() < SIZE {
            return None;
        }

        let mut pos = 0;
        buf[pos] = self.status;
        pos += 1;
        buf[pos..pos + 4].copy_from_slice(&self.raster_fd.to_le_bytes());
        buf[pos + 4..pos + 8].fill(0);
        pos += 8;
        buf[pos..pos + 4].copy_from_slice(&self.width.to_le_bytes());
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.height.to_le_bytes());
        pos += 4;
        buf[pos..pos + 4].copy_from_slice(&self.stride_bytes.to_le_bytes());
        pos += 4;
        buf[pos] = self.pixel_format;
        pos += 1;
        buf[pos..pos + 8].copy_from_slice(&self.variant_hash.to_le_bytes());
        pos += 8;

        Some(pos)
    }

    /// Decode response from wire format
    pub fn decode(buf: &[u8]) -> Option<Self> {
        const SIZE: usize = 1 + 8 + 4 + 4 + 4 + 1 + 8;
        if buf.len() < SIZE {
            return None;
        }

        let mut pos = 0;
        let status = buf[pos];
        pos += 1;
        let raster_fd = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
        pos += 8;
        let width = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let height = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let stride_bytes = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let pixel_format = buf[pos];
        pos += 1;
        let variant_hash = u64::from_le_bytes(buf[pos..pos + 8].try_into().ok()?);

        Some(Self {
            status,
            raster_fd,
            width,
            height,
            stride_bytes,
            pixel_format,
            variant_hash,
        })
    }
}

/// Decode request tag from buffer
pub fn decode_request_tag(buf: &[u8]) -> Option<SvgRequestTag> {
    if buf.is_empty() {
        return None;
    }
    match buf[0] {
        0 => Some(SvgRequestTag::Ping),
        1 => Some(SvgRequestTag::RasterizeSvg),
        _ => None,
    }
}

/// Encode an error response
pub fn encode_error(status: SvgStatus, buf: &mut [u8]) -> Option<usize> {
    let response = RasterizeSvgResponse {
        status: status as u8,
        raster_fd: 0,
        width: 0,
        height: 0,
        stride_bytes: 0,
        pixel_format: 0,
        variant_hash: 0,
    };
    response.encode(buf)
}

/// Encode a ping request
pub fn encode_ping(buf: &mut [u8]) -> Option<usize> {
    if buf.is_empty() {
        return None;
    }
    buf[0] = SvgRequestTag::Ping as u8;
    Some(1)
}
