use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C)]
pub struct FontId(pub u64); // Graph Node ID

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct FaceId {
    pub font_id: FontId,
    pub index: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontRequest {
    ListFonts,
    RenderText {
        face: FaceId,
        size_px: u32,
        text: alloc::string::String,
        color: u32, // ARGB
    },
    MeasureText {
        face: FaceId,
        size_px: u32,
        text: alloc::string::String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontInfo {
    pub face_id: FaceId,
    pub family: alloc::string::String,
    pub style: alloc::string::String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBitmap {
    pub width: u32,
    pub height: u32,
    pub baseline_y: i32,
    pub buffer_id: u64, // Bytespace ID containing raw pixels (ARGB or A8)
    pub buffer_size: usize,
    pub format_a8: bool, // true = A8, false = ARGB
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextMetrics {
    pub width: u32,
    pub height: u32,
    pub baseline_y: i32,
    pub advance_x: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontResponse {
    FontList(alloc::vec::Vec<FontInfo>),
    Rendered(TextBitmap),
    Measured(TextMetrics),
    Error(alloc::string::String),
}

impl FontRequest {
    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < 1 {
            return None;
        }
        match self {
            FontRequest::ListFonts => {
                buf[0] = 0;
                Some(1)
            }
            FontRequest::RenderText {
                face,
                size_px,
                text,
                color,
            } => {
                buf[0] = 1;
                let mut offset = 1;
                // face: FontId(u64) + index(u32) = 12 bytes
                if buf.len() < offset + 12 + 4 + 4 + 4 + text.len() {
                    return None;
                }
                buf[offset..offset + 8].copy_from_slice(&face.font_id.0.to_le_bytes());
                offset += 8;
                buf[offset..offset + 4].copy_from_slice(&face.index.to_le_bytes());
                offset += 4;
                buf[offset..offset + 4].copy_from_slice(&size_px.to_le_bytes());
                offset += 4;
                buf[offset..offset + 4].copy_from_slice(&color.to_le_bytes());
                offset += 4;
                let len = text.len() as u32;
                buf[offset..offset + 4].copy_from_slice(&len.to_le_bytes());
                offset += 4;
                buf[offset..offset + text.len()].copy_from_slice(text.as_bytes());
                offset += text.len();
                Some(offset)
            }
            FontRequest::MeasureText {
                face,
                size_px,
                text,
            } => {
                buf[0] = 2;
                let mut offset = 1;
                if buf.len() < offset + 12 + 4 + 4 + text.len() {
                    return None;
                }
                buf[offset..offset + 8].copy_from_slice(&face.font_id.0.to_le_bytes());
                offset += 8;
                buf[offset..offset + 4].copy_from_slice(&face.index.to_le_bytes());
                offset += 4;
                buf[offset..offset + 4].copy_from_slice(&size_px.to_le_bytes());
                offset += 4;
                let len = text.len() as u32;
                buf[offset..offset + 4].copy_from_slice(&len.to_le_bytes());
                offset += 4;
                buf[offset..offset + text.len()].copy_from_slice(text.as_bytes());
                offset += text.len();
                Some(offset)
            }
        }
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 1 {
            return None;
        }
        match buf[0] {
            0 => Some(FontRequest::ListFonts),
            1 => {
                let mut offset = 1;
                if buf.len() < offset + 20 {
                    return None;
                } // min header
                let font_id = u64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?);
                offset += 8;
                let index = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let size_px = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let color = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let len = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
                offset += 4;
                if buf.len() < offset + len {
                    return None;
                }
                let text = alloc::string::String::from_utf8(alloc::vec::Vec::from(
                    &buf[offset..offset + len],
                ))
                .ok()?;
                Some(FontRequest::RenderText {
                    face: FaceId {
                        font_id: FontId(font_id),
                        index,
                    },
                    size_px,
                    text,
                    color,
                })
            }
            2 => {
                let mut offset = 1;
                if buf.len() < offset + 16 {
                    return None;
                }
                let font_id = u64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?);
                offset += 8;
                let index = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let size_px = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let len = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
                offset += 4;
                if buf.len() < offset + len {
                    return None;
                }
                let text = alloc::string::String::from_utf8(alloc::vec::Vec::from(
                    &buf[offset..offset + len],
                ))
                .ok()?;
                Some(FontRequest::MeasureText {
                    face: FaceId {
                        font_id: FontId(font_id),
                        index,
                    },
                    size_px,
                    text,
                })
            }
            _ => None,
        }
    }
}

impl FontResponse {
    pub fn encode(&self, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < 1 {
            return None;
        }
        match self {
            FontResponse::FontList(list) => {
                buf[0] = 0;
                let mut offset = 1;
                let count = list.len() as u32;
                if buf.len() < offset + 4 {
                    return None;
                }
                buf[offset..offset + 4].copy_from_slice(&count.to_le_bytes());
                offset += 4;
                for item in list {
                    // font_id(8) + index(4) + family_len(4) + family + style_len(4) + style
                    let f_len = item.family.len();
                    let s_len = item.style.len();
                    if buf.len() < offset + 12 + 4 + f_len + 4 + s_len {
                        return None;
                    }
                    buf[offset..offset + 8].copy_from_slice(&item.face_id.font_id.0.to_le_bytes());
                    offset += 8;
                    buf[offset..offset + 4].copy_from_slice(&item.face_id.index.to_le_bytes());
                    offset += 4;
                    buf[offset..offset + 4].copy_from_slice(&(f_len as u32).to_le_bytes());
                    offset += 4;
                    buf[offset..offset + f_len].copy_from_slice(item.family.as_bytes());
                    offset += f_len;
                    buf[offset..offset + 4].copy_from_slice(&(s_len as u32).to_le_bytes());
                    offset += 4;
                    buf[offset..offset + s_len].copy_from_slice(item.style.as_bytes());
                    offset += s_len;
                }
                Some(offset)
            }
            FontResponse::Rendered(bmp) => {
                buf[0] = 1;
                let mut offset = 1;
                // w(4) + h(4) + base(4) + buf_id(8) + buf_size(8) + fmt(1)
                if buf.len() < offset + 29 {
                    return None;
                }
                buf[offset..offset + 4].copy_from_slice(&bmp.width.to_le_bytes());
                offset += 4;
                buf[offset..offset + 4].copy_from_slice(&bmp.height.to_le_bytes());
                offset += 4;
                buf[offset..offset + 4].copy_from_slice(&bmp.baseline_y.to_le_bytes());
                offset += 4;
                buf[offset..offset + 8].copy_from_slice(&bmp.buffer_id.to_le_bytes());
                offset += 8;
                buf[offset..offset + 8].copy_from_slice(&(bmp.buffer_size as u64).to_le_bytes());
                offset += 8;
                buf[offset] = if bmp.format_a8 { 1 } else { 0 };
                offset += 1;
                Some(offset)
            }
            FontResponse::Measured(m) => {
                buf[0] = 2;
                let mut offset = 1;
                if buf.len() < offset + 16 {
                    return None;
                }
                buf[offset..offset + 4].copy_from_slice(&m.width.to_le_bytes());
                offset += 4;
                buf[offset..offset + 4].copy_from_slice(&m.height.to_le_bytes());
                offset += 4;
                buf[offset..offset + 4].copy_from_slice(&m.baseline_y.to_le_bytes());
                offset += 4;
                buf[offset..offset + 4].copy_from_slice(&m.advance_x.to_le_bytes());
                offset += 4;
                Some(offset)
            }
            FontResponse::Error(msg) => {
                buf[0] = 3;
                let mut offset = 1;
                let len = msg.len() as u32;
                if buf.len() < offset + 4 + msg.len() {
                    return None;
                }
                buf[offset..offset + 4].copy_from_slice(&len.to_le_bytes());
                offset += 4;
                buf[offset..offset + msg.len()].copy_from_slice(msg.as_bytes());
                offset += msg.len();
                Some(offset)
            }
        }
    }

    pub fn decode(buf: &[u8]) -> Option<Self> {
        if buf.len() < 1 {
            return None;
        }
        match buf[0] {
            0 => {
                let mut offset = 1;
                if buf.len() < offset + 4 {
                    return None;
                }
                let count = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let mut list = alloc::vec::Vec::new();
                for _ in 0..count {
                    if buf.len() < offset + 20 {
                        return None;
                    }
                    let font_id = u64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?);
                    offset += 8;
                    let index = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                    offset += 4;
                    let f_len =
                        u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
                    offset += 4;
                    if buf.len() < offset + f_len + 4 {
                        return None;
                    }
                    let family = alloc::string::String::from_utf8(alloc::vec::Vec::from(
                        &buf[offset..offset + f_len],
                    ))
                    .ok()?;
                    offset += f_len;
                    let s_len =
                        u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
                    offset += 4;
                    if buf.len() < offset + s_len {
                        return None;
                    }
                    let style = alloc::string::String::from_utf8(alloc::vec::Vec::from(
                        &buf[offset..offset + s_len],
                    ))
                    .ok()?;
                    offset += s_len;
                    list.push(FontInfo {
                        face_id: FaceId {
                            font_id: FontId(font_id),
                            index,
                        },
                        family,
                        style,
                    });
                }
                Some(FontResponse::FontList(list))
            }
            1 => {
                let mut offset = 1;
                if buf.len() < offset + 29 {
                    return None;
                }
                let width = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let height = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let baseline_y = i32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let buffer_id = u64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?);
                offset += 8;
                let buffer_size =
                    u64::from_le_bytes(buf[offset..offset + 8].try_into().ok()?) as usize;
                offset += 8;
                let format_a8 = buf[offset] != 0;
                Some(FontResponse::Rendered(TextBitmap {
                    width,
                    height,
                    baseline_y,
                    buffer_id,
                    buffer_size,
                    format_a8,
                }))
            }
            2 => {
                let mut offset = 1;
                if buf.len() < offset + 16 {
                    return None;
                }
                let width = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let height = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let baseline_y = i32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                offset += 4;
                let advance_x = i32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?);
                Some(FontResponse::Measured(TextMetrics {
                    width,
                    height,
                    baseline_y,
                    advance_x,
                }))
            }
            3 => {
                let mut offset = 1;
                if buf.len() < offset + 4 {
                    return None;
                }
                let len = u32::from_le_bytes(buf[offset..offset + 4].try_into().ok()?) as usize;
                offset += 4;
                if buf.len() < offset + len {
                    return None;
                }
                let msg = alloc::string::String::from_utf8(alloc::vec::Vec::from(
                    &buf[offset..offset + len],
                ))
                .ok()?;
                Some(FontResponse::Error(msg))
            }
            _ => None,
        }
    }
}
