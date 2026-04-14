#![deny(missing_docs)]

//! Core definitions for ThingOS.
//!
//! This crate is intentionally small for now. It establishes a stable home in
//! the workspace for ThingOS-owned concepts while the fork continues to track
//! upstream Rust closely.

/// The canonical public name of the system.
pub const NAME: &str = "ThingOS";

/// The architectural premise that guides the system.
pub const TAGLINE: &str = "Types are the system itself.";

/// A coarse-grained classification for first-class objects in ThingOS.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KindClass {
    /// Any object in the system.
    Thing,
    /// A context in which Things exist and interact.
    Place,
    /// An actor that can act upon Things.
    Person,
}
