//! Message module: canonical `thingos.message` envelope integration.
//!
//! Re-exports [`thingos::message`] and provides a minimal kernel-side bridge
//! that allows other kernel subsystems to construct and inspect [`Message`]
//! values without importing the `thingos` crate directly.
//!
//! # Design intent
//!
//! `Message` is the **atom of communication** in ThingOS.  This module makes
//! the type available to all kernel subsystems so that no subsystem needs to
//! invent a bespoke event or notification struct.
//!
//! Delivery semantics (Inbox, Group broadcast, Port-based IPC) are *not*
//! implemented here; this module only establishes availability and provides
//! ergonomic construction helpers.
//!
//! See [`thingos::message`] for the full type documentation.

pub mod bridge;

/// Re-export of the canonical [`thingos::message::Message`] type.
pub use thingos::message::Message;

/// Re-export of the canonical [`thingos::message::KindId`] type.
pub use thingos::message::KindId;
