//! UI event wire format v1 — versioned, variable-length, graph-targeted.
//!
//! ## Wire layout
//!
//! ```text
//! ┌─────────┬─────────┬──────┬─────────────┬─────────────┐
//! │ magic   │ version │ kind │ payload_len │  payload …  │
//! │ u32 LE  │ u16 LE  │u16 LE│   u32 LE    │  variable   │
//! └─────────┴─────────┴──────┴─────────────┴─────────────┘
//!   4 bytes   2 bytes  2 bytes   4 bytes    payload_len
//! ```
//!
//! Unknown kinds can be skipped safely via `payload_len`.

/// Magic bytes: `"UIEV"` as little-endian u32.
pub const MAGIC: u32 = 0x5645_4955;

/// Current wire version.
pub const VERSION: u16 = 1;

/// Header size in bytes (magic + version + kind + payload_len).
pub const HEADER_SIZE: usize = 12;

/// Maximum UTF-8 text payload for `TextInput` events.
pub const TEXT_MAX: usize = 32;

// ── Legacy compat constants (deprecated) ─────────────────────────────
/// Old fixed event size. Kept briefly for migration reference.
#[deprecated(note = "Use HEADER_SIZE + per-event payload instead")]
pub const UI_EVENT_BYTES: usize = 64;

// ── Event kind discriminants ─────────────────────────────────────────

/// Numeric discriminant for each event kind (u16, wire-stable).
#[repr(u16)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UiEventKind {
    Focus = 1,
    Blur = 2,
    Activate = 3,
    Submit = 4,
    PointerMove = 5,
    PointerDown = 6,
    PointerUp = 7,
    Scroll = 8,
    KeyDown = 9,
    KeyUp = 10,
    TextInput = 11,
    TextBackspace = 12,
    TextDelete = 13,
    CursorMove = 14,
    CursorSet = 15,
    Select = 16,
    Resize = 17,
    CloseRequested = 18,
    Clicked = 19,
    Toggled = 20,
}

impl UiEventKind {
    pub fn from_raw(raw: u16) -> Option<Self> {
        match raw {
            1 => Some(Self::Focus),
            2 => Some(Self::Blur),
            3 => Some(Self::Activate),
            4 => Some(Self::Submit),
            5 => Some(Self::PointerMove),
            6 => Some(Self::PointerDown),
            7 => Some(Self::PointerUp),
            8 => Some(Self::Scroll),
            9 => Some(Self::KeyDown),
            10 => Some(Self::KeyUp),
            11 => Some(Self::TextInput),
            12 => Some(Self::TextBackspace),
            13 => Some(Self::TextDelete),
            14 => Some(Self::CursorMove),
            15 => Some(Self::CursorSet),
            16 => Some(Self::Select),
            17 => Some(Self::Resize),
            18 => Some(Self::CloseRequested),
            19 => Some(Self::Clicked),
            20 => Some(Self::Toggled),
            _ => None,
        }
    }
}

// ── High-level event enum ────────────────────────────────────────────

/// Decoded UI event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UiEvent {
    /// Set input focus to a target node.
    Focus { window: u64, target: u64 },
    /// Remove input focus from a target node.
    Blur { window: u64, target: u64 },
    /// Generic activation (click/space/enter on button-like thing).
    Activate { window: u64, target: u64 },
    /// Enter/submit for TextInput or forms.
    Submit { window: u64, target: u64 },

    /// Pointer moved (no target — global within window).
    PointerMove { window: u64, x: i32, y: i32 },
    /// Pointer button pressed on a target.
    PointerDown {
        window: u64,
        target: u64,
        button: u8,
        x: i32,
        y: i32,
        mods: u16,
    },
    /// Pointer button released on a target.
    PointerUp {
        window: u64,
        target: u64,
        button: u8,
        x: i32,
        y: i32,
        mods: u16,
    },
    /// Scroll wheel on a target.
    Scroll {
        window: u64,
        target: u64,
        dx: i32,
        dy: i32,
        mods: u16,
    },

    /// Raw key press.
    KeyDown { window: u64, key: u32, mods: u16 },
    /// Raw key release.
    KeyUp { window: u64, key: u32, mods: u16 },
    /// Insert text at cursor (composition result / printable chars).
    TextInput {
        window: u64,
        target: u64,
        text_len: u8,
        text: [u8; TEXT_MAX],
    },
    /// Delete one character before cursor.
    TextBackspace { window: u64, target: u64 },
    /// Delete one character after cursor.
    TextDelete { window: u64, target: u64 },
    /// Move cursor by delta characters.
    CursorMove {
        window: u64,
        target: u64,
        delta: i32,
    },
    /// Set cursor to absolute position.
    CursorSet { window: u64, target: u64, pos: u32 },
    /// Set text selection range.
    Select {
        window: u64,
        target: u64,
        start: u32,
        end: u32,
    },

    /// Window resized.
    Resize { window: u64, w: u32, h: u32 },
    /// Window close requested.
    CloseRequested { window: u64 },

    /// Legacy: button clicked.
    Clicked {
        window: u64,
        target: u64,
        action_id: u64,
    },
    /// Legacy: checkbox toggled.
    Toggled {
        window: u64,
        target: u64,
        checked: u8,
        value_id: u64,
    },

    /// An event kind we don't recognize (forward compat).
    Unknown { kind: u16, payload_len: u32 },
}

impl UiEvent {
    /// Convenience: get the event kind discriminant.
    pub fn kind(&self) -> Option<UiEventKind> {
        match self {
            Self::Focus { .. } => Some(UiEventKind::Focus),
            Self::Blur { .. } => Some(UiEventKind::Blur),
            Self::Activate { .. } => Some(UiEventKind::Activate),
            Self::Submit { .. } => Some(UiEventKind::Submit),
            Self::PointerMove { .. } => Some(UiEventKind::PointerMove),
            Self::PointerDown { .. } => Some(UiEventKind::PointerDown),
            Self::PointerUp { .. } => Some(UiEventKind::PointerUp),
            Self::Scroll { .. } => Some(UiEventKind::Scroll),
            Self::KeyDown { .. } => Some(UiEventKind::KeyDown),
            Self::KeyUp { .. } => Some(UiEventKind::KeyUp),
            Self::TextInput { .. } => Some(UiEventKind::TextInput),
            Self::TextBackspace { .. } => Some(UiEventKind::TextBackspace),
            Self::TextDelete { .. } => Some(UiEventKind::TextDelete),
            Self::CursorMove { .. } => Some(UiEventKind::CursorMove),
            Self::CursorSet { .. } => Some(UiEventKind::CursorSet),
            Self::Select { .. } => Some(UiEventKind::Select),
            Self::Resize { .. } => Some(UiEventKind::Resize),
            Self::CloseRequested { .. } => Some(UiEventKind::CloseRequested),
            Self::Clicked { .. } => Some(UiEventKind::Clicked),
            Self::Toggled { .. } => Some(UiEventKind::Toggled),
            Self::Unknown { .. } => None,
        }
    }

    /// Convenience: UTF-8 text bytes for TextInput events.
    pub fn text_bytes(&self) -> &[u8] {
        match self {
            Self::TextInput { text_len, text, .. } => &text[..*text_len as usize],
            _ => &[],
        }
    }
}

// ── Decode error ─────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Buffer too short for header or declared payload.
    Truncated,
    /// Magic bytes don't match.
    BadMagic,
    /// Version we can't handle.
    UnsupportedVersion,
}

// ── Encoder ──────────────────────────────────────────────────────────

fn write_header(out: &mut [u8], kind: u16, payload_len: u32) {
    out[0..4].copy_from_slice(&MAGIC.to_le_bytes());
    out[4..6].copy_from_slice(&VERSION.to_le_bytes());
    out[6..8].copy_from_slice(&kind.to_le_bytes());
    out[8..12].copy_from_slice(&payload_len.to_le_bytes());
}

fn put_u64(out: &mut [u8], offset: usize, v: u64) {
    out[offset..offset + 8].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(out: &mut [u8], offset: usize, v: u32) {
    out[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_i32(out: &mut [u8], offset: usize, v: i32) {
    out[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_u16(out: &mut [u8], offset: usize, v: u16) {
    out[offset..offset + 2].copy_from_slice(&v.to_le_bytes());
}

/// Payload size for a 2×u64 (window + target) event.
const P_PAIR: u32 = 16;

/// Encode a UI event into `out`. Returns `Some(bytes_written)` or `None` if
/// the buffer is too small.
pub fn encode(event: &UiEvent, out: &mut [u8]) -> Option<usize> {
    match event {
        // ── Focus / Blur / Activate / Submit (16-byte payload) ──
        UiEvent::Focus { window, target }
        | UiEvent::Blur { window, target }
        | UiEvent::Activate { window, target }
        | UiEvent::Submit { window, target }
        | UiEvent::TextBackspace { window, target }
        | UiEvent::TextDelete { window, target } => {
            let kind = match event {
                UiEvent::Focus { .. } => UiEventKind::Focus,
                UiEvent::Blur { .. } => UiEventKind::Blur,
                UiEvent::Activate { .. } => UiEventKind::Activate,
                UiEvent::Submit { .. } => UiEventKind::Submit,
                UiEvent::TextBackspace { .. } => UiEventKind::TextBackspace,
                UiEvent::TextDelete { .. } => UiEventKind::TextDelete,
                _ => unreachable!(),
            };
            let total = HEADER_SIZE + P_PAIR as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, kind as u16, P_PAIR);
            put_u64(out, HEADER_SIZE, *window);
            put_u64(out, HEADER_SIZE + 8, *target);
            Some(total)
        }

        // ── PointerMove (16 bytes: window + x + y + pad) ──
        UiEvent::PointerMove { window, x, y } => {
            let plen: u32 = 16;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, UiEventKind::PointerMove as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_i32(out, HEADER_SIZE + 8, *x);
            put_i32(out, HEADER_SIZE + 12, *y);
            Some(total)
        }

        // ── PointerDown / PointerUp (28 bytes) ──
        UiEvent::PointerDown {
            window,
            target,
            button,
            x,
            y,
            mods,
        }
        | UiEvent::PointerUp {
            window,
            target,
            button,
            x,
            y,
            mods,
        } => {
            let kind = if matches!(event, UiEvent::PointerDown { .. }) {
                UiEventKind::PointerDown
            } else {
                UiEventKind::PointerUp
            };
            let plen: u32 = 28;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, kind as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_u64(out, HEADER_SIZE + 8, *target);
            out[HEADER_SIZE + 16] = *button;
            out[HEADER_SIZE + 17] = 0; // pad
            put_u16(out, HEADER_SIZE + 18, *mods);
            put_i32(out, HEADER_SIZE + 20, *x);
            put_i32(out, HEADER_SIZE + 24, *y);
            Some(total)
        }

        // ── Scroll (28 bytes) ──
        UiEvent::Scroll {
            window,
            target,
            dx,
            dy,
            mods,
        } => {
            let plen: u32 = 28;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, UiEventKind::Scroll as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_u64(out, HEADER_SIZE + 8, *target);
            put_i32(out, HEADER_SIZE + 16, *dx);
            put_i32(out, HEADER_SIZE + 20, *dy);
            put_u16(out, HEADER_SIZE + 24, *mods);
            out[HEADER_SIZE + 26] = 0; // pad
            out[HEADER_SIZE + 27] = 0;
            Some(total)
        }

        // ── KeyDown / KeyUp (14 bytes → pad to 16) ──
        UiEvent::KeyDown { window, key, mods } | UiEvent::KeyUp { window, key, mods } => {
            let kind = if matches!(event, UiEvent::KeyDown { .. }) {
                UiEventKind::KeyDown
            } else {
                UiEventKind::KeyUp
            };
            let plen: u32 = 16;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, kind as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_u32(out, HEADER_SIZE + 8, *key);
            put_u16(out, HEADER_SIZE + 12, *mods);
            out[HEADER_SIZE + 14] = 0;
            out[HEADER_SIZE + 15] = 0;
            Some(total)
        }

        // ── TextInput (16 + 1 + 1 + text bytes, padded) ──
        UiEvent::TextInput {
            window,
            target,
            text_len,
            text,
        } => {
            let tlen = core::cmp::min(*text_len as usize, TEXT_MAX);
            // payload = window(8) + target(8) + text_len(1) + pad(1) + text(tlen)
            let raw_plen = 8 + 8 + 1 + 1 + tlen;
            // Round up to 4 for alignment.
            let plen = ((raw_plen + 3) / 4) * 4;
            let total = HEADER_SIZE + plen;
            if out.len() < total {
                return None;
            }
            // Zero padding area.
            for b in out[HEADER_SIZE..HEADER_SIZE + plen].iter_mut() {
                *b = 0;
            }
            write_header(out, UiEventKind::TextInput as u16, plen as u32);
            put_u64(out, HEADER_SIZE, *window);
            put_u64(out, HEADER_SIZE + 8, *target);
            out[HEADER_SIZE + 16] = tlen as u8;
            out[HEADER_SIZE + 17] = 0;
            out[HEADER_SIZE + 18..HEADER_SIZE + 18 + tlen].copy_from_slice(&text[..tlen]);
            Some(total)
        }

        // ── CursorMove (20 bytes) ──
        UiEvent::CursorMove {
            window,
            target,
            delta,
        } => {
            let plen: u32 = 20;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, UiEventKind::CursorMove as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_u64(out, HEADER_SIZE + 8, *target);
            put_i32(out, HEADER_SIZE + 16, *delta);
            Some(total)
        }

        // ── CursorSet (20 bytes) ──
        UiEvent::CursorSet {
            window,
            target,
            pos,
        } => {
            let plen: u32 = 20;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, UiEventKind::CursorSet as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_u64(out, HEADER_SIZE + 8, *target);
            put_u32(out, HEADER_SIZE + 16, *pos);
            Some(total)
        }

        // ── Select (24 bytes) ──
        UiEvent::Select {
            window,
            target,
            start,
            end,
        } => {
            let plen: u32 = 24;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, UiEventKind::Select as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_u64(out, HEADER_SIZE + 8, *target);
            put_u32(out, HEADER_SIZE + 16, *start);
            put_u32(out, HEADER_SIZE + 20, *end);
            Some(total)
        }

        // ── Resize (16 bytes: window + w + h) ──
        UiEvent::Resize { window, w, h } => {
            let plen: u32 = 16;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, UiEventKind::Resize as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_u32(out, HEADER_SIZE + 8, *w);
            put_u32(out, HEADER_SIZE + 12, *h);
            Some(total)
        }

        // ── CloseRequested (8 bytes: window only) ──
        UiEvent::CloseRequested { window } => {
            let plen: u32 = 8;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, UiEventKind::CloseRequested as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            Some(total)
        }

        // ── Clicked (24 bytes) ──
        UiEvent::Clicked {
            window,
            target,
            action_id,
        } => {
            let plen: u32 = 24;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, UiEventKind::Clicked as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_u64(out, HEADER_SIZE + 8, *target);
            put_u64(out, HEADER_SIZE + 16, *action_id);
            Some(total)
        }

        // ── Toggled (28 bytes) ──
        UiEvent::Toggled {
            window,
            target,
            checked,
            value_id,
        } => {
            let plen: u32 = 24;
            let total = HEADER_SIZE + plen as usize;
            if out.len() < total {
                return None;
            }
            write_header(out, UiEventKind::Toggled as u16, plen);
            put_u64(out, HEADER_SIZE, *window);
            put_u64(out, HEADER_SIZE + 8, *target);
            out[HEADER_SIZE + 16] = *checked;
            out[HEADER_SIZE + 17] = 0;
            out[HEADER_SIZE + 18] = 0;
            out[HEADER_SIZE + 19] = 0;
            put_u32(out, HEADER_SIZE + 20, (*value_id & 0xFFFF_FFFF) as u32);
            Some(total)
        }

        // ── Unknown (cannot be encoded by callers) ──
        UiEvent::Unknown { .. } => None,
    }
}

// ── Decoder ──────────────────────────────────────────────────────────

fn get_u64(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(buf[offset..offset + 8].try_into().unwrap())
}

fn get_u32(buf: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn get_i32(buf: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap())
}

fn get_u16(buf: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(buf[offset..offset + 2].try_into().unwrap())
}

/// Decode one event from the front of `bytes`.
///
/// Returns `Ok((event, consumed))` where `consumed` is the total bytes
/// (header + payload) to advance past. Unknown event kinds are returned as
/// `UiEvent::Unknown` so callers can skip gracefully.
pub fn decode_one(bytes: &[u8]) -> Result<(UiEvent, usize), DecodeError> {
    if bytes.len() < HEADER_SIZE {
        return Err(DecodeError::Truncated);
    }

    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    if magic != MAGIC {
        return Err(DecodeError::BadMagic);
    }

    let version = get_u16(bytes, 4);
    if version == 0 || version > VERSION {
        return Err(DecodeError::UnsupportedVersion);
    }

    let kind_raw = get_u16(bytes, 6);
    let payload_len = get_u32(bytes, 8) as usize;
    let total = HEADER_SIZE + payload_len;

    if bytes.len() < total {
        return Err(DecodeError::Truncated);
    }

    let p = &bytes[HEADER_SIZE..total];

    let event = match UiEventKind::from_raw(kind_raw) {
        Some(UiEventKind::Focus) if payload_len >= 16 => UiEvent::Focus {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
        },
        Some(UiEventKind::Blur) if payload_len >= 16 => UiEvent::Blur {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
        },
        Some(UiEventKind::Activate) if payload_len >= 16 => UiEvent::Activate {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
        },
        Some(UiEventKind::Submit) if payload_len >= 16 => UiEvent::Submit {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
        },
        Some(UiEventKind::TextBackspace) if payload_len >= 16 => UiEvent::TextBackspace {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
        },
        Some(UiEventKind::TextDelete) if payload_len >= 16 => UiEvent::TextDelete {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
        },

        Some(UiEventKind::PointerMove) if payload_len >= 16 => UiEvent::PointerMove {
            window: get_u64(p, 0),
            x: get_i32(p, 8),
            y: get_i32(p, 12),
        },

        Some(UiEventKind::PointerDown) if payload_len >= 28 => UiEvent::PointerDown {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
            button: p[16],
            mods: get_u16(p, 18),
            x: get_i32(p, 20),
            y: get_i32(p, 24),
        },
        Some(UiEventKind::PointerUp) if payload_len >= 28 => UiEvent::PointerUp {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
            button: p[16],
            mods: get_u16(p, 18),
            x: get_i32(p, 20),
            y: get_i32(p, 24),
        },

        Some(UiEventKind::Scroll) if payload_len >= 28 => UiEvent::Scroll {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
            dx: get_i32(p, 16),
            dy: get_i32(p, 20),
            mods: get_u16(p, 24),
        },

        Some(UiEventKind::KeyDown) if payload_len >= 16 => UiEvent::KeyDown {
            window: get_u64(p, 0),
            key: get_u32(p, 8),
            mods: get_u16(p, 12),
        },
        Some(UiEventKind::KeyUp) if payload_len >= 16 => UiEvent::KeyUp {
            window: get_u64(p, 0),
            key: get_u32(p, 8),
            mods: get_u16(p, 12),
        },

        Some(UiEventKind::TextInput) if payload_len >= 18 => {
            let tlen = core::cmp::min(p[16] as usize, TEXT_MAX);
            let mut text = [0u8; TEXT_MAX];
            let avail = core::cmp::min(tlen, payload_len - 18);
            text[..avail].copy_from_slice(&p[18..18 + avail]);
            UiEvent::TextInput {
                window: get_u64(p, 0),
                target: get_u64(p, 8),
                text_len: avail as u8,
                text,
            }
        }

        Some(UiEventKind::CursorMove) if payload_len >= 20 => UiEvent::CursorMove {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
            delta: get_i32(p, 16),
        },

        Some(UiEventKind::CursorSet) if payload_len >= 20 => UiEvent::CursorSet {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
            pos: get_u32(p, 16),
        },

        Some(UiEventKind::Select) if payload_len >= 24 => UiEvent::Select {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
            start: get_u32(p, 16),
            end: get_u32(p, 20),
        },

        Some(UiEventKind::Resize) if payload_len >= 16 => UiEvent::Resize {
            window: get_u64(p, 0),
            w: get_u32(p, 8),
            h: get_u32(p, 12),
        },

        Some(UiEventKind::CloseRequested) if payload_len >= 8 => UiEvent::CloseRequested {
            window: get_u64(p, 0),
        },

        Some(UiEventKind::Clicked) if payload_len >= 24 => UiEvent::Clicked {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
            action_id: get_u64(p, 16),
        },

        Some(UiEventKind::Toggled) if payload_len >= 24 => UiEvent::Toggled {
            window: get_u64(p, 0),
            target: get_u64(p, 8),
            checked: p[16],
            value_id: get_u32(p, 20) as u64,
        },

        // Unknown kind OR known kind with too-short payload — skip safely.
        _ => UiEvent::Unknown {
            kind: kind_raw,
            payload_len: payload_len as u32,
        },
    };

    Ok((event, total))
}

// ── Convenience constructors ─────────────────────────────────────────

impl UiEvent {
    pub fn focus(window: u64, target: u64) -> Self {
        Self::Focus { window, target }
    }
    pub fn blur(window: u64, target: u64) -> Self {
        Self::Blur { window, target }
    }
    pub fn activate(window: u64, target: u64) -> Self {
        Self::Activate { window, target }
    }
    pub fn submit(window: u64, target: u64) -> Self {
        Self::Submit { window, target }
    }
    pub fn text_backspace(window: u64, target: u64) -> Self {
        Self::TextBackspace { window, target }
    }
    pub fn text_delete(window: u64, target: u64) -> Self {
        Self::TextDelete { window, target }
    }
    pub fn text_input(window: u64, target: u64, bytes: &[u8]) -> Self {
        let len = core::cmp::min(bytes.len(), TEXT_MAX);
        let mut text = [0u8; TEXT_MAX];
        text[..len].copy_from_slice(&bytes[..len]);
        Self::TextInput {
            window,
            target,
            text_len: len as u8,
            text,
        }
    }
    pub fn cursor_move(window: u64, target: u64, delta: i32) -> Self {
        Self::CursorMove {
            window,
            target,
            delta,
        }
    }
    pub fn clicked(window: u64, target: u64, action_id: u64) -> Self {
        Self::Clicked {
            window,
            target,
            action_id,
        }
    }
    pub fn toggled(window: u64, target: u64, checked: bool, value_id: u64) -> Self {
        Self::Toggled {
            window,
            target,
            checked: if checked { 1 } else { 0 },
            value_id,
        }
    }
}

// ── Legacy shim ──────────────────────────────────────────────────────

/// Deprecated compat shim. Prefer `UiEvent` + `encode`/`decode_one` directly.
#[deprecated(note = "Use UiEvent + ui_event::encode / ui_event::decode_one")]
#[derive(Clone, Debug)]
pub struct UiEventWire {
    pub event: UiEvent,
}

#[allow(deprecated)]
impl UiEventWire {
    pub fn new_clicked(window_id: u64, target_id: u64, action_id: u64) -> Self {
        Self {
            event: UiEvent::clicked(window_id, target_id, action_id),
        }
    }
    pub fn new_toggled(window_id: u64, target_id: u64, checked: bool, value_id: u64) -> Self {
        Self {
            event: UiEvent::toggled(window_id, target_id, checked, value_id),
        }
    }
    pub fn new_focus(window_id: u64, target_id: u64) -> Self {
        Self {
            event: UiEvent::focus(window_id, target_id),
        }
    }
    pub fn new_blur(window_id: u64, target_id: u64) -> Self {
        Self {
            event: UiEvent::blur(window_id, target_id),
        }
    }
    pub fn new_text_insert(window_id: u64, target_id: u64, text: &[u8]) -> Self {
        Self {
            event: UiEvent::text_input(window_id, target_id, text),
        }
    }
    pub fn new_text_backspace(window_id: u64, target_id: u64) -> Self {
        Self {
            event: UiEvent::text_backspace(window_id, target_id),
        }
    }
    pub fn new_text_delete(window_id: u64, target_id: u64) -> Self {
        Self {
            event: UiEvent::text_delete(window_id, target_id),
        }
    }
    pub fn new_cursor_move(window_id: u64, target_id: u64, delta: i32) -> Self {
        Self {
            event: UiEvent::cursor_move(window_id, target_id, delta),
        }
    }
    pub fn new_submit(window_id: u64, target_id: u64) -> Self {
        Self {
            event: UiEvent::submit(window_id, target_id),
        }
    }
    pub fn encode(&self, out: &mut [u8]) -> Option<usize> {
        encode(&self.event, out)
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(event: UiEvent) {
        let mut buf = [0u8; 256];
        let written = encode(&event, &mut buf).expect("encode failed");
        let (decoded, consumed) = decode_one(&buf[..written]).expect("decode failed");
        assert_eq!(consumed, written);
        assert_eq!(decoded, event);
    }

    #[test]
    fn roundtrip_focus() {
        roundtrip(UiEvent::focus(42, 99));
    }

    #[test]
    fn roundtrip_blur() {
        roundtrip(UiEvent::blur(1, 2));
    }

    #[test]
    fn roundtrip_activate() {
        roundtrip(UiEvent::activate(1, 2));
    }

    #[test]
    fn roundtrip_submit() {
        roundtrip(UiEvent::submit(10, 20));
    }

    #[test]
    fn roundtrip_pointer_move() {
        roundtrip(UiEvent::PointerMove {
            window: 5,
            x: -100,
            y: 200,
        });
    }

    #[test]
    fn roundtrip_pointer_down() {
        roundtrip(UiEvent::PointerDown {
            window: 1,
            target: 2,
            button: 0,
            x: 10,
            y: 20,
            mods: 0x03,
        });
    }

    #[test]
    fn roundtrip_pointer_up() {
        roundtrip(UiEvent::PointerUp {
            window: 1,
            target: 2,
            button: 1,
            x: 30,
            y: 40,
            mods: 0,
        });
    }

    #[test]
    fn roundtrip_scroll() {
        roundtrip(UiEvent::Scroll {
            window: 7,
            target: 8,
            dx: -3,
            dy: 5,
            mods: 0,
        });
    }

    #[test]
    fn roundtrip_key_down() {
        roundtrip(UiEvent::KeyDown {
            window: 1,
            key: 0x41,
            mods: 0x01,
        });
    }

    #[test]
    fn roundtrip_key_up() {
        roundtrip(UiEvent::KeyUp {
            window: 1,
            key: 0x41,
            mods: 0,
        });
    }

    #[test]
    fn roundtrip_text_input() {
        roundtrip(UiEvent::text_input(1, 2, b"hello"));
    }

    #[test]
    fn roundtrip_text_backspace() {
        roundtrip(UiEvent::text_backspace(1, 2));
    }

    #[test]
    fn roundtrip_text_delete() {
        roundtrip(UiEvent::text_delete(1, 2));
    }

    #[test]
    fn roundtrip_cursor_move() {
        roundtrip(UiEvent::cursor_move(1, 2, -1));
    }

    #[test]
    fn roundtrip_cursor_set() {
        roundtrip(UiEvent::CursorSet {
            window: 1,
            target: 2,
            pos: 42,
        });
    }

    #[test]
    fn roundtrip_select() {
        roundtrip(UiEvent::Select {
            window: 1,
            target: 2,
            start: 3,
            end: 10,
        });
    }

    #[test]
    fn roundtrip_resize() {
        roundtrip(UiEvent::Resize {
            window: 1,
            w: 800,
            h: 600,
        });
    }

    #[test]
    fn roundtrip_close_requested() {
        roundtrip(UiEvent::CloseRequested { window: 1 });
    }

    #[test]
    fn roundtrip_clicked() {
        roundtrip(UiEvent::clicked(1, 2, 42));
    }

    #[test]
    fn roundtrip_toggled() {
        roundtrip(UiEvent::toggled(1, 2, true, 7));
    }

    #[test]
    fn unknown_kind_skipped() {
        let mut buf = [0u8; 64];
        // Write a fake header with unknown kind=999 and a 4-byte payload.
        buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        buf[4..6].copy_from_slice(&VERSION.to_le_bytes());
        buf[6..8].copy_from_slice(&999u16.to_le_bytes());
        buf[8..12].copy_from_slice(&4u32.to_le_bytes());
        // 4 bytes of dummy payload.
        buf[12..16].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let (event, consumed) = decode_one(&buf).unwrap();
        assert_eq!(consumed, 16);
        assert_eq!(
            event,
            UiEvent::Unknown {
                kind: 999,
                payload_len: 4
            }
        );
    }

    #[test]
    fn truncated_header_fails() {
        let buf = [0u8; 8]; // less than HEADER_SIZE
        assert_eq!(decode_one(&buf), Err(DecodeError::Truncated));
    }

    #[test]
    fn truncated_payload_fails() {
        let mut buf = [0u8; 14]; // header + 2, but claims 16
        buf[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        buf[4..6].copy_from_slice(&VERSION.to_le_bytes());
        buf[6..8].copy_from_slice(&(UiEventKind::Focus as u16).to_le_bytes());
        buf[8..12].copy_from_slice(&16u32.to_le_bytes());
        assert_eq!(decode_one(&buf), Err(DecodeError::Truncated));
    }

    #[test]
    fn bad_magic_fails() {
        let mut buf = [0u8; 32];
        buf[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        buf[4..6].copy_from_slice(&VERSION.to_le_bytes());
        buf[6..8].copy_from_slice(&1u16.to_le_bytes());
        buf[8..12].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(decode_one(&buf), Err(DecodeError::BadMagic));
    }

    #[test]
    fn multi_event_sequential_decode() {
        let mut buf = [0u8; 512];
        let mut offset = 0;

        let e1 = UiEvent::focus(1, 10);
        offset += encode(&e1, &mut buf[offset..]).unwrap();

        let e2 = UiEvent::text_input(1, 10, b"abc");
        offset += encode(&e2, &mut buf[offset..]).unwrap();

        let e3 = UiEvent::text_backspace(1, 10);
        offset += encode(&e3, &mut buf[offset..]).unwrap();

        // Decode all three sequentially.
        let mut pos = 0;
        let (d1, c1) = decode_one(&buf[pos..offset]).unwrap();
        pos += c1;
        assert_eq!(d1, e1);

        let (d2, c2) = decode_one(&buf[pos..offset]).unwrap();
        pos += c2;
        assert_eq!(d2, e2);

        let (d3, c3) = decode_one(&buf[pos..offset]).unwrap();
        pos += c3;
        assert_eq!(d3, e3);

        assert_eq!(pos, offset);
    }
}
