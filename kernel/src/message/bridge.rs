//! Bridge layer: kernel raw bytes → canonical `thingos.message::Message`.
//!
//! # Purpose
//!
//! This module is the **single construction point** for turning raw kernel
//! IPC data into a canonical [`Message`].  It mirrors the bridge pattern used
//! by [`crate::task::bridge`], [`crate::job::bridge`], and similar modules.
//!
//! # Constraints
//!
//! - No delivery or routing logic belongs here.
//! - No sender, reply-to, or sequencing metadata is added.
//! - Keep construction helpers minimal and stable.

use thingos::message::{KindId, Message};

/// Construct a [`Message`] from a raw `kind_id` byte array and an owned
/// payload buffer.
///
/// This is the preferred kernel-internal entry point for building a canonical
/// `Message` from subsystem-local data.  The `kind_id` bytes are typically a
/// `KIND_ID_*` constant emitted by `kindc`.
pub fn message_from_parts(kind: KindId, payload: alloc::vec::Vec<u8>) -> Message {
    Message::new(kind, payload)
}

/// Construct an empty (no-payload) [`Message`] for the given kind.
///
/// Useful for pure-signal notifications where the `kind` alone carries the
/// semantics and no additional data is needed.
pub fn message_signal(kind: KindId) -> Message {
    Message::new(kind, alloc::vec::Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_from_parts_round_trips() {
        let kind = KindId([0xabu8; 16]);
        let payload = alloc::vec![1u8, 2, 3];
        let msg = message_from_parts(kind, payload.clone());
        assert_eq!(msg.kind, kind);
        assert_eq!(msg.payload, payload);
    }

    #[test]
    fn test_message_signal_is_empty() {
        let kind = KindId::THINGOS_MESSAGE;
        let msg = message_signal(kind);
        assert!(msg.is_empty());
        assert_eq!(msg.kind, kind);
    }
}
