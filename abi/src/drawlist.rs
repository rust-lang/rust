#![allow(clippy::needless_range_loop)]

extern crate alloc;

use alloc::vec::Vec;

/// Packed drawlist wire format for VFS-native rendering.
///
/// # Layout
/// ```text
/// Header:
///   magic: u32 (DLST)
///   version: u16
///   reserved: u16
///   cmd_count: u32
///   reserved2: u32
///
/// Commands:
///   tag: u32
///   len: u32
///   payload: [u8; len]
/// ```
///
/// Commands are simple TLVs so apps can append without per-command watches.
///
/// # Example
/// ```
/// use abi::drawlist::{DrawListBuilder, FillRule, PathVerb, PointF};
///
/// let mut builder = DrawListBuilder::new();
/// builder.push_save();
/// builder.push_set_clip_rect(0, 0, 100, 100);
/// builder.push_fill_rect(10, 10, 20, 30, 0xff00ff00);
/// let path = vec![
///     PathVerb::MoveTo(PointF::new(0.0, 0.0)),
///     PathVerb::LineTo(PointF::new(10.0, 0.0)),
///     PathVerb::LineTo(PointF::new(10.0, 10.0)),
///     PathVerb::Close,
/// ];
/// builder.push_fill_path(&path, FillRule::NonZero, 0xffff0000);
/// builder.push_restore();
/// let bytes = builder.finish();
/// assert!(bytes.len() > 16);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum DrawListVersion {
    V1 = 1,
}

pub const DRAWLIST_MAGIC: u32 = 0x4453_4c54; // "DLST"
const DRAWLIST_HEADER_BYTES: usize = 16;
const DRAWLIST_CMD_COUNT_OFFSET: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DrawCmdTag {
    FillRect = 1,
    FillPath = 2,
    Line = 3,
    StrokePath = 4,
    TextSpan = 5,
    DrawIcon = 6,
    Save = 7,
    Restore = 8,
    SetClipRect = 9,
    SetTransform = 10,
    DrawImageRect = 11,
    Unknown(u32),
}

impl DrawCmdTag {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            1 => DrawCmdTag::FillRect,
            2 => DrawCmdTag::FillPath,
            3 => DrawCmdTag::Line,
            4 => DrawCmdTag::StrokePath,
            5 => DrawCmdTag::TextSpan,
            6 => DrawCmdTag::DrawIcon,
            7 => DrawCmdTag::Save,
            8 => DrawCmdTag::Restore,
            9 => DrawCmdTag::SetClipRect,
            10 => DrawCmdTag::SetTransform,
            11 => DrawCmdTag::DrawImageRect,
            _ => DrawCmdTag::Unknown(raw),
        }
    }

    pub fn as_raw(self) -> u32 {
        match self {
            DrawCmdTag::FillRect => 1,
            DrawCmdTag::FillPath => 2,
            DrawCmdTag::Line => 3,
            DrawCmdTag::StrokePath => 4,
            DrawCmdTag::TextSpan => 5,
            DrawCmdTag::DrawIcon => 6,
            DrawCmdTag::Save => 7,
            DrawCmdTag::Restore => 8,
            DrawCmdTag::SetClipRect => 9,
            DrawCmdTag::SetTransform => 10,
            DrawCmdTag::DrawImageRect => 11,
            DrawCmdTag::Unknown(raw) => raw,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FillRule {
    NonZero = 0,
    EvenOdd = 1,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointF {
    pub x: f32,
    pub y: f32,
}

impl PointF {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathVerb {
    MoveTo(PointF),
    LineTo(PointF),
    QuadTo(PointF, PointF),
    CubicTo(PointF, PointF, PointF),
    Close,
}

pub struct DrawListBuilder {
    bytes: Vec<u8>,
    cmd_count: u32,
}

impl DrawListBuilder {
    pub fn new() -> Self {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&DRAWLIST_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&(DrawListVersion::V1 as u16).to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        debug_assert_eq!(bytes.len(), DRAWLIST_HEADER_BYTES);
        Self {
            bytes,
            cmd_count: 0,
        }
    }

    pub fn push_fill_rect(&mut self, x: i32, y: i32, w: i32, h: i32, color: u32) {
        let mut payload = Vec::with_capacity(20);
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&w.to_le_bytes());
        payload.extend_from_slice(&h.to_le_bytes());
        payload.extend_from_slice(&color.to_le_bytes());
        self.push_cmd(DrawCmdTag::FillRect, &payload);
    }

    pub fn push_fill_path(&mut self, verbs: &[PathVerb], fill_rule: FillRule, color: u32) {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(fill_rule as u32).to_le_bytes());
        payload.extend_from_slice(&color.to_le_bytes());
        let path_bytes = encode_path(verbs);
        payload.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(&path_bytes);
        self.push_cmd(DrawCmdTag::FillPath, &payload);
    }

    pub fn push_line(&mut self, from: PointF, to: PointF, color: u32, width: f32) {
        let mut payload = Vec::with_capacity(24);
        payload.extend_from_slice(&from.x.to_le_bytes());
        payload.extend_from_slice(&from.y.to_le_bytes());
        payload.extend_from_slice(&to.x.to_le_bytes());
        payload.extend_from_slice(&to.y.to_le_bytes());
        payload.extend_from_slice(&color.to_le_bytes());
        payload.extend_from_slice(&width.to_le_bytes());
        self.push_cmd(DrawCmdTag::Line, &payload);
    }

    pub fn push_stroke_path(&mut self, verbs: &[PathVerb], width: f32, color: u32) {
        let mut payload = Vec::new();
        payload.extend_from_slice(&width.to_le_bytes());
        payload.extend_from_slice(&color.to_le_bytes());
        let path_bytes = encode_path(verbs);
        payload.extend_from_slice(&(path_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(&path_bytes);
        self.push_cmd(DrawCmdTag::StrokePath, &payload);
    }

    pub fn push_text_span(&mut self, text: &str, x: f32, y: f32, size: f32, color: u32) {
        let mut payload = Vec::new();
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&size.to_le_bytes());
        payload.extend_from_slice(&color.to_le_bytes());
        let text_bytes = text.as_bytes();
        payload.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
        payload.extend_from_slice(text_bytes);
        self.push_cmd(DrawCmdTag::TextSpan, &payload);
    }

    pub fn push_draw_icon(&mut self, x: i32, y: i32, w: i32, h: i32, icon_id: u32) {
        let mut payload = Vec::with_capacity(20);
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&w.to_le_bytes());
        payload.extend_from_slice(&h.to_le_bytes());
        payload.extend_from_slice(&icon_id.to_le_bytes());
        self.push_cmd(DrawCmdTag::DrawIcon, &payload);
    }

    /// Save the current graphics state (transform, clip).
    pub fn push_save(&mut self) {
        self.push_cmd(DrawCmdTag::Save, &[]);
    }

    /// Restore the previously saved graphics state.
    pub fn push_restore(&mut self) {
        self.push_cmd(DrawCmdTag::Restore, &[]);
    }

    /// Set a rectangular clipping region.
    ///
    /// **Important**: This command should be used within a Save/Restore pair
    /// to ensure proper state management. The renderer will push a new clip
    /// onto the clip stack, which must be balanced with a Restore.
    ///
    /// # Example
    /// ```
    /// # use abi::drawlist::DrawListBuilder;
    /// # let mut builder = DrawListBuilder::new();
    /// builder.push_save();
    /// builder.push_set_clip_rect(0, 0, 100, 100);
    /// // ... draw commands ...
    /// builder.push_restore();
    /// ```
    pub fn push_set_clip_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        let mut payload = Vec::with_capacity(16);
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
        payload.extend_from_slice(&w.to_le_bytes());
        payload.extend_from_slice(&h.to_le_bytes());
        self.push_cmd(DrawCmdTag::SetClipRect, &payload);
    }

    /// Set an affine 2D transform matrix [a, b, c, d, tx, ty].
    /// The transform maps points as: x' = a*x + c*y + tx, y' = b*x + d*y + ty
    ///
    /// **Important**: This command should be used within a Save/Restore pair
    /// to ensure proper state management. The renderer will push a new transform
    /// onto the transform stack, which must be balanced with a Restore.
    ///
    /// # Example
    /// ```
    /// # use abi::drawlist::DrawListBuilder;
    /// # let mut builder = DrawListBuilder::new();
    /// builder.push_save();
    /// builder.push_set_transform(2.0, 0.0, 0.0, 2.0, 0.0, 0.0); // 2x scale
    /// // ... draw commands ...
    /// builder.push_restore();
    /// ```
    pub fn push_set_transform(&mut self, a: f32, b: f32, c: f32, d: f32, tx: f32, ty: f32) {
        let mut payload = Vec::with_capacity(24);
        payload.extend_from_slice(&a.to_le_bytes());
        payload.extend_from_slice(&b.to_le_bytes());
        payload.extend_from_slice(&c.to_le_bytes());
        payload.extend_from_slice(&d.to_le_bytes());
        payload.extend_from_slice(&tx.to_le_bytes());
        payload.extend_from_slice(&ty.to_le_bytes());
        self.push_cmd(DrawCmdTag::SetTransform, &payload);
    }

    /// Draw an image from a graph ThingId into a destination rectangle.
    /// image_id: ThingId (as u128) of the image asset.
    pub fn push_draw_image_rect(
        &mut self,
        image_id: u128,
        src_x: i32,
        src_y: i32,
        src_w: i32,
        src_h: i32,
        dst_x: i32,
        dst_y: i32,
        dst_w: i32,
        dst_h: i32,
    ) {
        let mut payload = Vec::with_capacity(48);
        payload.extend_from_slice(&image_id.to_le_bytes());
        payload.extend_from_slice(&src_x.to_le_bytes());
        payload.extend_from_slice(&src_y.to_le_bytes());
        payload.extend_from_slice(&src_w.to_le_bytes());
        payload.extend_from_slice(&src_h.to_le_bytes());
        payload.extend_from_slice(&dst_x.to_le_bytes());
        payload.extend_from_slice(&dst_y.to_le_bytes());
        payload.extend_from_slice(&dst_w.to_le_bytes());
        payload.extend_from_slice(&dst_h.to_le_bytes());
        self.push_cmd(DrawCmdTag::DrawImageRect, &payload);
    }

    pub fn finish(mut self) -> Vec<u8> {
        let cmd_count_bytes = self.cmd_count.to_le_bytes();
        self.bytes[DRAWLIST_CMD_COUNT_OFFSET..DRAWLIST_CMD_COUNT_OFFSET + 4]
            .copy_from_slice(&cmd_count_bytes);
        self.bytes
    }

    fn push_cmd(&mut self, tag: DrawCmdTag, payload: &[u8]) {
        self.bytes.extend_from_slice(&tag.as_raw().to_le_bytes());
        self.bytes
            .extend_from_slice(&(payload.len() as u32).to_le_bytes());
        self.bytes.extend_from_slice(payload);
        self.cmd_count += 1;
    }
}

#[derive(Debug, Clone)]
pub struct DrawCmdRef<'a> {
    pub tag: DrawCmdTag,
    pub payload: &'a [u8],
}

pub struct DrawListReader<'a> {
    bytes: &'a [u8],
    offset: usize,
    remaining: u32,
}

impl<'a> DrawListReader<'a> {
    pub fn new(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().ok()?);
        if magic != DRAWLIST_MAGIC {
            return None;
        }
        let version = u16::from_le_bytes(bytes[4..6].try_into().ok()?);
        if version != DrawListVersion::V1 as u16 {
            return None;
        }
        let cmd_count = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
        Some(Self {
            bytes,
            offset: 16,
            remaining: cmd_count,
        })
    }

    /// Returns the next known command in the stream, skipping unknown tags.
    pub fn next(&mut self) -> Option<DrawCmdRef<'a>> {
        while self.remaining > 0 {
            if self.offset + 8 > self.bytes.len() {
                self.remaining = 0;
                return None;
            }
            let raw_tag =
                u32::from_le_bytes(self.bytes[self.offset..self.offset + 4].try_into().ok()?);
            let len = u32::from_le_bytes(
                self.bytes[self.offset + 4..self.offset + 8]
                    .try_into()
                    .ok()?,
            );
            self.offset += 8;
            let end = self.offset + len as usize;
            if end > self.bytes.len() {
                self.remaining = 0;
                return None;
            }
            let payload = &self.bytes[self.offset..end];
            self.offset = end;
            self.remaining -= 1;
            let tag = DrawCmdTag::from_raw(raw_tag);
            if matches!(tag, DrawCmdTag::Unknown(_)) {
                continue;
            }
            return Some(DrawCmdRef { tag, payload });
        }
        None
    }

    /// Validate that the drawlist stream is well-formed and matches cmd_count.
    pub fn validate(bytes: &'a [u8]) -> Result<(), DrawListError> {
        let mut reader = DrawListReader::new(bytes).ok_or(DrawListError::InvalidHeader)?;
        while reader.remaining > 0 {
            if reader.offset + 8 > reader.bytes.len() {
                return Err(DrawListError::TruncatedCommand);
            }
            let len = u32::from_le_bytes(
                reader.bytes[reader.offset + 4..reader.offset + 8]
                    .try_into()
                    .map_err(|_| DrawListError::TruncatedCommand)?,
            );
            reader.offset += 8;
            let end = reader
                .offset
                .checked_add(len as usize)
                .ok_or(DrawListError::TruncatedCommand)?;
            if end > reader.bytes.len() {
                return Err(DrawListError::TruncatedCommand);
            }
            reader.offset = end;
            reader.remaining -= 1;
        }
        if reader.offset == reader.bytes.len() {
            Ok(())
        } else {
            Err(DrawListError::TrailingBytes)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawListError {
    InvalidHeader,
    TruncatedCommand,
    TrailingBytes,
}

pub fn decode_fill_rect(payload: &[u8]) -> Option<(i32, i32, i32, i32, u32)> {
    if payload.len() != 20 {
        return None;
    }
    let x = i32::from_le_bytes(payload[0..4].try_into().ok()?);
    let y = i32::from_le_bytes(payload[4..8].try_into().ok()?);
    let w = i32::from_le_bytes(payload[8..12].try_into().ok()?);
    let h = i32::from_le_bytes(payload[12..16].try_into().ok()?);
    let color = u32::from_le_bytes(payload[16..20].try_into().ok()?);
    Some((x, y, w, h, color))
}

pub struct DecodedFillPath {
    pub fill_rule: FillRule,
    pub color: u32,
    pub verbs: Vec<PathVerb>,
}

pub fn decode_fill_path(payload: &[u8]) -> Option<DecodedFillPath> {
    if payload.len() < 12 {
        return None;
    }
    // Strict format: payload must match header + path bytes exactly.
    let fill_rule = match u32::from_le_bytes(payload[0..4].try_into().ok()?) {
        0 => FillRule::NonZero,
        1 => FillRule::EvenOdd,
        _ => return None,
    };
    let color = u32::from_le_bytes(payload[4..8].try_into().ok()?);
    let path_len = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
    if payload.len() != 12 + path_len {
        return None;
    }
    let verbs = decode_path(&payload[12..12 + path_len])?;
    Some(DecodedFillPath {
        fill_rule,
        color,
        verbs,
    })
}

pub fn decode_line(payload: &[u8]) -> Option<(PointF, PointF, u32, f32)> {
    if payload.len() != 24 {
        return None;
    }
    let fx = f32::from_le_bytes(payload[0..4].try_into().ok()?);
    let fy = f32::from_le_bytes(payload[4..8].try_into().ok()?);
    let tx = f32::from_le_bytes(payload[8..12].try_into().ok()?);
    let ty = f32::from_le_bytes(payload[12..16].try_into().ok()?);
    let color = u32::from_le_bytes(payload[16..20].try_into().ok()?);
    let width = f32::from_le_bytes(payload[20..24].try_into().ok()?);
    Some((PointF::new(fx, fy), PointF::new(tx, ty), color, width))
}

pub struct DecodedStrokePath {
    pub width: f32,
    pub color: u32,
    pub verbs: Vec<PathVerb>,
}

pub fn decode_stroke_path(payload: &[u8]) -> Option<DecodedStrokePath> {
    if payload.len() < 12 {
        return None;
    }
    let width = f32::from_le_bytes(payload[0..4].try_into().ok()?);
    let color = u32::from_le_bytes(payload[4..8].try_into().ok()?);
    let path_len = u32::from_le_bytes(payload[8..12].try_into().ok()?) as usize;
    if payload.len() != 12 + path_len {
        return None;
    }
    let verbs = decode_path(&payload[12..12 + path_len])?;
    Some(DecodedStrokePath {
        width,
        color,
        verbs,
    })
}

pub struct DecodedTextSpan {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub color: u32,
    pub text: alloc::string::String,
}

pub fn decode_text_span(payload: &[u8]) -> Option<DecodedTextSpan> {
    if payload.len() < 20 {
        return None;
    }
    let x = f32::from_le_bytes(payload[0..4].try_into().ok()?);
    let y = f32::from_le_bytes(payload[4..8].try_into().ok()?);
    let size = f32::from_le_bytes(payload[8..12].try_into().ok()?);
    let color = u32::from_le_bytes(payload[12..16].try_into().ok()?);
    let text_len = u32::from_le_bytes(payload[16..20].try_into().ok()?) as usize;
    if payload.len() != 20 + text_len {
        return None;
    }
    let text = core::str::from_utf8(&payload[20..20 + text_len])
        .ok()?
        .into();
    Some(DecodedTextSpan {
        x,
        y,
        size,
        color,
        text,
    })
}

pub fn decode_draw_icon(payload: &[u8]) -> Option<(i32, i32, i32, i32, u32)> {
    if payload.len() != 20 {
        return None;
    }
    let x = i32::from_le_bytes(payload[0..4].try_into().ok()?);
    let y = i32::from_le_bytes(payload[4..8].try_into().ok()?);
    let w = i32::from_le_bytes(payload[8..12].try_into().ok()?);
    let h = i32::from_le_bytes(payload[12..16].try_into().ok()?);
    let icon_id = u32::from_le_bytes(payload[16..20].try_into().ok()?);
    Some((x, y, w, h, icon_id))
}

/// Decode Save command (no payload).
pub fn decode_save(payload: &[u8]) -> Option<()> {
    if payload.is_empty() {
        Some(())
    } else {
        None
    }
}

/// Decode Restore command (no payload).
pub fn decode_restore(payload: &[u8]) -> Option<()> {
    if payload.is_empty() {
        Some(())
    } else {
        None
    }
}

/// Decode SetClipRect command.
pub fn decode_set_clip_rect(payload: &[u8]) -> Option<(i32, i32, i32, i32)> {
    if payload.len() != 16 {
        return None;
    }
    let x = i32::from_le_bytes(payload[0..4].try_into().ok()?);
    let y = i32::from_le_bytes(payload[4..8].try_into().ok()?);
    let w = i32::from_le_bytes(payload[8..12].try_into().ok()?);
    let h = i32::from_le_bytes(payload[12..16].try_into().ok()?);
    Some((x, y, w, h))
}

/// Affine 2D transform matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform2D {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub tx: f32,
    pub ty: f32,
}

/// Decode SetTransform command.
pub fn decode_set_transform(payload: &[u8]) -> Option<Transform2D> {
    if payload.len() != 24 {
        return None;
    }
    let a = f32::from_le_bytes(payload[0..4].try_into().ok()?);
    let b = f32::from_le_bytes(payload[4..8].try_into().ok()?);
    let c = f32::from_le_bytes(payload[8..12].try_into().ok()?);
    let d = f32::from_le_bytes(payload[12..16].try_into().ok()?);
    let tx = f32::from_le_bytes(payload[16..20].try_into().ok()?);
    let ty = f32::from_le_bytes(payload[20..24].try_into().ok()?);
    Some(Transform2D { a, b, c, d, tx, ty })
}

/// Decoded DrawImageRect command.
pub struct DecodedDrawImageRect {
    pub image_id: u128,
    pub src_x: i32,
    pub src_y: i32,
    pub src_w: i32,
    pub src_h: i32,
    pub dst_x: i32,
    pub dst_y: i32,
    pub dst_w: i32,
    pub dst_h: i32,
}

/// Decode DrawImageRect command.
pub fn decode_draw_image_rect(payload: &[u8]) -> Option<DecodedDrawImageRect> {
    if payload.len() != 48 {
        return None;
    }
    let image_id = u128::from_le_bytes(payload[0..16].try_into().ok()?);
    let src_x = i32::from_le_bytes(payload[16..20].try_into().ok()?);
    let src_y = i32::from_le_bytes(payload[20..24].try_into().ok()?);
    let src_w = i32::from_le_bytes(payload[24..28].try_into().ok()?);
    let src_h = i32::from_le_bytes(payload[28..32].try_into().ok()?);
    let dst_x = i32::from_le_bytes(payload[32..36].try_into().ok()?);
    let dst_y = i32::from_le_bytes(payload[36..40].try_into().ok()?);
    let dst_w = i32::from_le_bytes(payload[40..44].try_into().ok()?);
    let dst_h = i32::from_le_bytes(payload[44..48].try_into().ok()?);
    Some(DecodedDrawImageRect {
        image_id,
        src_x,
        src_y,
        src_w,
        src_h,
        dst_x,
        dst_y,
        dst_w,
        dst_h,
    })
}

fn encode_path(verbs: &[PathVerb]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut verb_bytes = Vec::new();
    let mut points = Vec::new();
    for verb in verbs {
        match *verb {
            PathVerb::MoveTo(p) => {
                verb_bytes.push(0u8);
                points.push(p);
            }
            PathVerb::LineTo(p) => {
                verb_bytes.push(1u8);
                points.push(p);
            }
            PathVerb::QuadTo(c, p) => {
                verb_bytes.push(2u8);
                points.push(c);
                points.push(p);
            }
            PathVerb::CubicTo(c1, c2, p) => {
                verb_bytes.push(3u8);
                points.push(c1);
                points.push(c2);
                points.push(p);
            }
            PathVerb::Close => {
                verb_bytes.push(4u8);
            }
        }
    }
    out.extend_from_slice(&(verb_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&(points.len() as u32).to_le_bytes());
    out.extend_from_slice(&verb_bytes);
    for p in points {
        out.extend_from_slice(&p.x.to_le_bytes());
        out.extend_from_slice(&p.y.to_le_bytes());
    }
    out
}

fn decode_path(bytes: &[u8]) -> Option<Vec<PathVerb>> {
    if bytes.len() < 8 {
        return None;
    }
    let verb_len = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    let point_len = u32::from_le_bytes(bytes[4..8].try_into().ok()?) as usize;
    let verb_start = 8;
    let point_start = verb_start + verb_len;
    let point_bytes = point_len * 8;
    if bytes.len() != point_start + point_bytes {
        return None;
    }
    let verb_bytes = &bytes[verb_start..point_start];
    let points_bytes = &bytes[point_start..point_start + point_bytes];
    let mut points = Vec::with_capacity(point_len);
    for idx in 0..point_len {
        let base = idx * 8;
        let x = f32::from_le_bytes(points_bytes[base..base + 4].try_into().ok()?);
        let y = f32::from_le_bytes(points_bytes[base + 4..base + 8].try_into().ok()?);
        points.push(PointF { x, y });
    }
    let mut point_cursor = 0usize;
    let mut verbs = Vec::new();
    for verb in verb_bytes {
        let verb = match verb {
            0 => {
                let p = points.get(point_cursor)?;
                point_cursor += 1;
                PathVerb::MoveTo(*p)
            }
            1 => {
                let p = points.get(point_cursor)?;
                point_cursor += 1;
                PathVerb::LineTo(*p)
            }
            2 => {
                let c = points.get(point_cursor)?;
                let p = points.get(point_cursor + 1)?;
                point_cursor += 2;
                PathVerb::QuadTo(*c, *p)
            }
            3 => {
                let c1 = points.get(point_cursor)?;
                let c2 = points.get(point_cursor + 1)?;
                let p = points.get(point_cursor + 2)?;
                point_cursor += 3;
                PathVerb::CubicTo(*c1, *c2, *p)
            }
            4 => PathVerb::Close,
            _ => return None,
        };
        verbs.push(verb);
    }
    if point_cursor != point_len {
        return None;
    }
    Some(verbs)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    #[test]
    fn reader_skips_unknown_tags() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&DRAWLIST_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&(DrawListVersion::V1 as u16).to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&99u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3, 4]);
        let mut builder = DrawListBuilder::new();
        builder.push_fill_rect(1, 2, 3, 4, 0xff00ff00);
        let payload = builder.finish();
        bytes.extend_from_slice(&payload[16..]);

        let mut reader = DrawListReader::new(&bytes).expect("reader");
        let cmd = reader.next().expect("cmd");
        assert_eq!(cmd.tag, DrawCmdTag::FillRect);
        assert!(reader.next().is_none());
    }

    #[test]
    fn fill_path_rejects_trailing_bytes() {
        let mut builder = DrawListBuilder::new();
        let verbs = [PathVerb::MoveTo(PointF::new(0.0, 0.0))];
        builder.push_fill_path(&verbs, FillRule::NonZero, 0xff00ff00);
        let mut bytes = builder.finish();

        // Tamper with the command length to include the extra byte
        // Header (16) + Tag (4) = 20
        let len_offset = 20;
        let mut len = u32::from_le_bytes(bytes[len_offset..len_offset + 4].try_into().unwrap());
        len += 1;
        bytes[len_offset..len_offset + 4].copy_from_slice(&len.to_le_bytes());

        bytes.push(0);
        let mut reader = DrawListReader::new(&bytes).expect("reader");
        let cmd = reader.next().expect("cmd");
        assert!(decode_fill_path(cmd.payload).is_none());
    }

    #[test]
    fn path_rejects_trailing_bytes() {
        let verbs = [PathVerb::MoveTo(PointF::new(0.0, 0.0))];
        let mut payload = encode_path(&verbs);
        payload.push(0);
        assert!(decode_path(&payload).is_none());
    }

    #[test]
    fn validate_rejects_truncated_payload() {
        let mut builder = DrawListBuilder::new();
        builder.push_fill_rect(1, 2, 3, 4, 0xff00ff00);
        let mut bytes = builder.finish();
        bytes.truncate(bytes.len() - 2);
        assert_eq!(
            DrawListReader::validate(&bytes),
            Err(DrawListError::TruncatedCommand)
        );
    }

    #[test]
    fn validate_rejects_bad_cmd_count() {
        let mut builder = DrawListBuilder::new();
        builder.push_fill_rect(1, 2, 3, 4, 0xff00ff00);
        let mut bytes = builder.finish();
        bytes[DRAWLIST_CMD_COUNT_OFFSET..DRAWLIST_CMD_COUNT_OFFSET + 4]
            .copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            DrawListReader::validate(&bytes),
            Err(DrawListError::TruncatedCommand)
        );
    }

    #[test]
    fn validate_rejects_trailing_bytes() {
        let mut builder = DrawListBuilder::new();
        builder.push_fill_rect(1, 2, 3, 4, 0xff00ff00);
        let mut bytes = builder.finish();
        bytes.push(0);
        assert_eq!(
            DrawListReader::validate(&bytes),
            Err(DrawListError::TrailingBytes)
        );
    }

    #[test]
    fn save_restore_round_trip() {
        let mut builder = DrawListBuilder::new();
        builder.push_save();
        builder.push_restore();
        let bytes = builder.finish();

        let mut reader = DrawListReader::new(&bytes).expect("reader");
        let cmd1 = reader.next().expect("save");
        assert_eq!(cmd1.tag, DrawCmdTag::Save);
        assert!(decode_save(cmd1.payload).is_some());

        let cmd2 = reader.next().expect("restore");
        assert_eq!(cmd2.tag, DrawCmdTag::Restore);
        assert!(decode_restore(cmd2.payload).is_some());
        assert!(reader.next().is_none());
    }

    #[test]
    fn set_clip_rect_round_trip() {
        let mut builder = DrawListBuilder::new();
        builder.push_set_clip_rect(10, 20, 100, 200);
        let bytes = builder.finish();

        let mut reader = DrawListReader::new(&bytes).expect("reader");
        let cmd = reader.next().expect("clip");
        assert_eq!(cmd.tag, DrawCmdTag::SetClipRect);
        let (x, y, w, h) = decode_set_clip_rect(cmd.payload).expect("decode");
        assert_eq!((x, y, w, h), (10, 20, 100, 200));
    }

    #[test]
    fn set_transform_round_trip() {
        let mut builder = DrawListBuilder::new();
        builder.push_set_transform(1.0, 0.0, 0.0, 1.0, 50.0, 100.0);
        let bytes = builder.finish();

        let mut reader = DrawListReader::new(&bytes).expect("reader");
        let cmd = reader.next().expect("transform");
        assert_eq!(cmd.tag, DrawCmdTag::SetTransform);
        let t = decode_set_transform(cmd.payload).expect("decode");
        assert_eq!(t.a, 1.0);
        assert_eq!(t.b, 0.0);
        assert_eq!(t.c, 0.0);
        assert_eq!(t.d, 1.0);
        assert_eq!(t.tx, 50.0);
        assert_eq!(t.ty, 100.0);
    }

    #[test]
    fn draw_image_rect_round_trip() {
        let mut builder = DrawListBuilder::new();
        let image_id = 0x1234567890ABCDEF1234567890ABCDEFu128;
        builder.push_draw_image_rect(image_id, 0, 0, 100, 100, 10, 20, 80, 60);
        let bytes = builder.finish();

        let mut reader = DrawListReader::new(&bytes).expect("reader");
        let cmd = reader.next().expect("image");
        assert_eq!(cmd.tag, DrawCmdTag::DrawImageRect);
        let img = decode_draw_image_rect(cmd.payload).expect("decode");
        assert_eq!(img.image_id, image_id);
        assert_eq!(img.src_x, 0);
        assert_eq!(img.src_y, 0);
        assert_eq!(img.src_w, 100);
        assert_eq!(img.src_h, 100);
        assert_eq!(img.dst_x, 10);
        assert_eq!(img.dst_y, 20);
        assert_eq!(img.dst_w, 80);
        assert_eq!(img.dst_h, 60);
    }

    #[test]
    fn complex_drawlist_with_new_commands() {
        let mut builder = DrawListBuilder::new();
        builder.push_save();
        builder.push_set_clip_rect(0, 0, 200, 200);
        builder.push_set_transform(0.8, 0.0, 0.0, 0.8, 20.0, 20.0);
        builder.push_fill_rect(10, 10, 50, 50, 0xff00ff00);
        builder.push_line(
            PointF::new(0.0, 0.0),
            PointF::new(100.0, 100.0),
            0xffff0000,
            2.0,
        );
        builder.push_restore();
        let bytes = builder.finish();

        assert!(DrawListReader::validate(&bytes).is_ok());
        let mut reader = DrawListReader::new(&bytes).expect("reader");
        let mut count = 0;
        while reader.next().is_some() {
            count += 1;
        }
        assert_eq!(count, 6);
    }
}
