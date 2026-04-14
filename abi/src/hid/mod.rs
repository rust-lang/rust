//! Bristle HID Wire Protocol
//!
//! Types shared between HID drivers, Bristle broker, and consumer apps.
//! All types are `repr(C)` for wire compatibility.
//!
//! # Examples
//! ```
//! use abi::hid::{
//!     BristleEventHeader, EventType, Key, KeyEventPayload, Mods,
//!     BRISTLE_EVENT_MAGIC, BRISTLE_EVENT_VERSION,
//! };
//!
//! let header = BristleEventHeader {
//!     magic: BRISTLE_EVENT_MAGIC,
//!     version: BRISTLE_EVENT_VERSION,
//!     event_type: EventType::KeyDown as u16,
//!     timestamp_ns: 42,
//!     payload_len: KeyEventPayload::SIZE as u32,
//! };
//! let payload = KeyEventPayload {
//!     key: Key::A as u16,
//!     mods: Mods::SHIFT,
//!     flags: 0,
//! };
//!
//! let mut bytes = Vec::new();
//! bytes.extend_from_slice(&header.to_bytes());
//! bytes.extend_from_slice(&payload.to_bytes());
//!
//! let header_back = BristleEventHeader::from_bytes(
//!     bytes[..BristleEventHeader::SIZE].try_into().unwrap()
//! ).unwrap();
//! let payload_back = KeyEventPayload::from_bytes(
//!     bytes[BristleEventHeader::SIZE..].try_into().unwrap()
//! );
//!
//! let event_type = header_back.event_type;
//! assert_eq!(event_type, EventType::KeyDown as u16);
//! assert_eq!(payload_back.key(), Key::A);
//! assert!(payload_back.mods().has_shift());
//! ```

/// Magic number for Bristle event headers: 'HIDE'
pub const BRISTLE_EVENT_MAGIC: u32 = 0x48494445;

/// Protocol version
pub const BRISTLE_EVENT_VERSION: u16 = 0;

pub use bristle::{BristleEventHeader, KeyEventPayload};
pub use event::{EventType, HidParseError};
pub use input::{InputDeviceKind, Ps2KeyPayload, RawInputEnvelope};
pub use key::Key;
pub use modifiers::{Locks, Mods};
pub use pointer::{PointerButtonPayload, PointerMovePayload, ScrollPayload};

mod bristle;
mod event;
mod input;
mod key;
mod modifiers;
mod pointer;

#[cfg(test)]
mod tests;
