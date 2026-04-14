#![no_std]
#![deny(missing_docs)]
//! Core definitions for ThingOS.
//!
//! This crate is intentionally small for now. It establishes a stable home in
//! the workspace for ThingOS-owned concepts while the fork continues to track
//! upstream Rust closely.
//!
//! # Schema-generated canonical types
//!
//! The [`task`], [`job`], and [`group`] modules contain the schema-generated
//! public types that define the canonical external representation of execution,
//! lifecycle, and coordination concepts respectively.  The kernel's internal
//! `Thread` and `Process` structures are *transitional* implementations; they
//! feed into these public types through explicit bridge layers rather than
//! being exposed directly.
//!
//! ## Transitional mapping (Phase 1 – 4)
//!
//! | Canonical concept      | Current internal backing              | Future direction             |
//! |------------------------|---------------------------------------|------------------------------|
//! | `task::Task`           | kernel `Thread`                       | becomes the Task impl        |
//! | `job::Job`             | kernel `Process` (partial)            | hollowed out by further phases|
//! | `job::JobExit`         | `Thread::exit_code`                   | Phase 3 exit path            |
//! | `job::JobWaitResult`   | `poll_task_exit` result               | Phase 3 wait path            |
//! | `group::Group`         | `Process::pgid` / `ConsoleTtyState`   | Phase 4 coordination domain  |
//!
//! Public truth changes first; internal machinery follows.

extern crate alloc;

pub mod group;
pub mod job;
pub mod task;

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
