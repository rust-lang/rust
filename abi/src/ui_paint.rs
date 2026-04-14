extern crate alloc;

use alloc::vec::Vec;

pub const UI_PAINT_MAGIC: u32 = 0x5549_504e; // "UIPN"
pub const UI_PAINT_VERSION: u16 = 1;
const UI_PAINT_HEADER_BYTES: usize = 16;
const UI_PAINT_CMD_COUNT_OFFSET: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum PaintOpTag {
    PushClip = 1,
    PopClip = 2,
    FillRect = 3,
    DrawTextRun = 4,
    BlitImage = 5,
    StrokeLine = 6,
    DrawIcon = 7,
    FillLinearGradient = 8,
    Unknown(u32),
}

impl PaintOpTag {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            1 => PaintOpTag::PushClip,
            2 => PaintOpTag::PopClip,
            3 => PaintOpTag::FillRect,
            4 => PaintOpTag::DrawTextRun,
            5 => PaintOpTag::BlitImage,
            6 => PaintOpTag::StrokeLine,
            7 => PaintOpTag::DrawIcon,
            8 => PaintOpTag::FillLinearGradient,
            _ => PaintOpTag::Unknown(raw),
        }
    }

    pub fn as_raw(self) -> u32 {
        match self {
            PaintOpTag::PushClip => 1,
            PaintOpTag::PopClip => 2,
            PaintOpTag::FillRect => 3,
            PaintOpTag::DrawTextRun => 4,
            PaintOpTag::BlitImage => 5,
            PaintOpTag::StrokeLine => 6,
            PaintOpTag::DrawIcon => 7,
            PaintOpTag::FillLinearGradient => 8,
            PaintOpTag::Unknown(raw) => raw,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ImageFit {
    Fill = 0,
    Contain = 1,
    Cover = 2,
    None = 3,
}

impl ImageFit {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            1 => ImageFit::Contain,
            2 => ImageFit::Cover,
            3 => ImageFit::None,
            _ => ImageFit::Fill,
        }
    }
}

pub struct PaintBuilder {
    bytes: Vec<u8>,
    cmd_count: u32,
}

impl PaintBuilder {
    pub fn new() -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&UI_PAINT_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&UI_PAINT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        debug_assert_eq!(bytes.len(), UI_PAINT_HEADER_BYTES);
        Self {
            bytes,
            cmd_count: 0,
        }
    }

    pub fn push_clip(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let mut payload = Vec::with_capacity(16);
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&w.to_le_bytes());
        payload.extend_from_slice(&h.to_le_bytes());
        self.push_cmd(PaintOpTag::PushClip, &payload);
    }

    pub fn pop_clip(&mut self) {
        self.push_cmd(PaintOpTag::PopClip, &[]);
    }

    pub fn fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32) {
        let mut payload = Vec::with_capacity(20);
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&w.to_le_bytes());
        payload.extend_from_slice(&h.to_le_bytes());
        payload.extend_from_slice(&color.to_le_bytes());
        self.push_cmd(PaintOpTag::FillRect, &payload);
    }

    pub fn draw_text_run(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        baseline: i32,
        font_key: &str,
        size: i32,
        text: &str,
        color: u32,
    ) {
        let mut payload = Vec::new();
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&w.to_le_bytes());
        payload.extend_from_slice(&h.to_le_bytes());
        payload.extend_from_slice(&baseline.to_le_bytes());
        payload.extend_from_slice(&size.to_le_bytes());
        payload.extend_from_slice(&color.to_le_bytes());
        let font_bytes = font_key.as_bytes();
        let text_bytes = text.as_bytes();
        payload.extend_from_slice(&(font_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(font_bytes);
        payload.extend_from_slice(text_bytes);
        self.push_cmd(PaintOpTag::DrawTextRun, &payload);
    }

    pub fn blit_image(&mut self, x: i32, y: i32, w: i32, h: i32, fit: ImageFit, image_key: &str) {
        let mut payload = Vec::new();
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&w.to_le_bytes());
        payload.extend_from_slice(&h.to_le_bytes());
        payload.push(fit as u8);
        payload.extend_from_slice(&[0u8; 3]);
        let key_bytes = image_key.as_bytes();
        payload.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(key_bytes);
        self.push_cmd(PaintOpTag::BlitImage, &payload);
    }

    pub fn stroke_line(&mut self, x1: i32, y1: i32, x2: i32, y2: i32, width: i32, color: u32) {
        let mut payload = Vec::with_capacity(24);
        payload.extend_from_slice(&x1.to_le_bytes());
        payload.extend_from_slice(&y1.to_le_bytes());
        payload.extend_from_slice(&x2.to_le_bytes());
        payload.extend_from_slice(&y2.to_le_bytes());
        payload.extend_from_slice(&width.to_le_bytes());
        payload.extend_from_slice(&color.to_le_bytes());
        self.push_cmd(PaintOpTag::StrokeLine, &payload);
    }

    pub fn draw_icon(&mut self, x: i32, y: i32, w: i32, h: i32, name: &str) {
        let mut payload = Vec::new();
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&w.to_le_bytes());
        payload.extend_from_slice(&h.to_le_bytes());
        let name_bytes = name.as_bytes();
        payload.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(name_bytes);
        self.push_cmd(PaintOpTag::DrawIcon, &payload);
    }

    pub fn fill_linear_gradient(
        &mut self,
        x: i32,
        y: i32,
        w: i32,
        h: i32,
        color1: u32,
        color2: u32,
    ) {
        let mut payload = Vec::with_capacity(24);
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&w.to_le_bytes());
        payload.extend_from_slice(&h.to_le_bytes());
        payload.extend_from_slice(&color1.to_le_bytes());
        payload.extend_from_slice(&color2.to_le_bytes());
        self.push_cmd(PaintOpTag::FillLinearGradient, &payload);
    }

    pub fn finish(mut self) -> Vec<u8> {
        let cmd_count_bytes = self.cmd_count.to_le_bytes();
        self.bytes[UI_PAINT_CMD_COUNT_OFFSET..UI_PAINT_CMD_COUNT_OFFSET + 4]
            .copy_from_slice(&cmd_count_bytes);
        self.bytes
    }

    fn push_cmd(&mut self, tag: PaintOpTag, payload: &[u8]) {
        self.bytes.extend_from_slice(&tag.as_raw().to_le_bytes());
        self.bytes
            .extend_from_slice(&(payload.len() as u32).to_le_bytes());
        self.bytes.extend_from_slice(payload);
        self.cmd_count = self.cmd_count.saturating_add(1);
    }
}

#[derive(Debug, Clone)]
pub struct PaintOpRef<'a> {
    pub tag: PaintOpTag,
    pub payload: &'a [u8],
}

pub struct PaintReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
    remaining: u32,
}

impl<'a> PaintReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < UI_PAINT_HEADER_BYTES {
            return None;
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        if magic != UI_PAINT_MAGIC {
            return None;
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().ok()?);
        if version != UI_PAINT_VERSION {
            return None;
        }
        let cmd_count = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        Some(Self {
            bytes,
            cursor: UI_PAINT_HEADER_BYTES,
            remaining: cmd_count,
        })
    }
}

impl<'a> Iterator for PaintReader<'a> {
    type Item = PaintOpRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        if self.cursor + 8 > self.bytes.len() {
            return None;
        }
        let tag_raw = u32::from_le_bytes(self.bytes[self.cursor..self.cursor + 4].try_into().ok()?);
        let len = u32::from_le_bytes(
            self.bytes[self.cursor + 4..self.cursor + 8]
                .try_into()
                .ok()?,
        );
        let payload_start = self.cursor + 8;
        let payload_end = payload_start + len as usize;
        if payload_end > self.bytes.len() {
            return None;
        }
        self.cursor = payload_end;
        self.remaining -= 1;
        Some(PaintOpRef {
            tag: PaintOpTag::from_raw(tag_raw),
            payload: &self.bytes[payload_start..payload_end],
        })
    }
}

pub fn decode_fill_rect(payload: &[u8]) -> Option<(i32, i32, i32, i32, u32)> {
    if payload.len() < 20 {
        return None;
    }
    let x = i32::from_le_bytes(payload[0..4].try_into().ok()?);
    let y = i32::from_le_bytes(payload[4..8].try_into().ok()?);
    let w = i32::from_le_bytes(payload[8..12].try_into().ok()?);
    let h = i32::from_le_bytes(payload[12..16].try_into().ok()?);
    let color = u32::from_le_bytes(payload[16..20].try_into().ok()?);
    Some((x, y, w, h, color))
}

pub fn decode_fill_linear_gradient(payload: &[u8]) -> Option<(i32, i32, i32, i32, u32, u32)> {
    if payload.len() < 24 {
        return None;
    }
    let x = i32::from_le_bytes(payload[0..4].try_into().ok()?);
    let y = i32::from_le_bytes(payload[4..8].try_into().ok()?);
    let w = i32::from_le_bytes(payload[8..12].try_into().ok()?);
    let h = i32::from_le_bytes(payload[12..16].try_into().ok()?);
    let color1 = u32::from_le_bytes(payload[16..20].try_into().ok()?);
    let color2 = u32::from_le_bytes(payload[20..24].try_into().ok()?);
    Some((x, y, w, h, color1, color2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paint_builder_roundtrip() {
        let mut b = PaintBuilder::new();
        b.fill_rect(1, 2, 3, 4, 0xFF00FF00);
        b.pop_clip();
        let bytes = b.finish();
        let mut reader = PaintReader::new(&bytes).expect("reader");
        let first = reader.next().expect("first");
        assert_eq!(first.tag, PaintOpTag::FillRect);
        let second = reader.next().expect("second");
        assert_eq!(second.tag, PaintOpTag::PopClip);
    }
}
