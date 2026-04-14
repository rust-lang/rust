#![no_std]
#![deny(missing_docs)]
//! Core definitions for ThingOS.
//!
//! This crate is `no_std` so it can be used in both kernel and userspace
//! contexts without pulling in the host operating system's standard library.
//! Platform capabilities are accessed explicitly through `stem::pal` and the
//! built-in `core` + `alloc` crates.
//!
//! This crate is intentionally small for now. It establishes a stable home in
//! the workspace for ThingOS-owned concepts while the fork continues to track
//! upstream Rust closely.
//!
//! # Schema-generated canonical types
//!
//! The [`task`], [`job`], [`group`], and [`authority`] modules contain the
//! schema-generated public types that define the canonical external
//! representation of execution, lifecycle, coordination, and permission-context
//! concepts respectively.  The kernel's internal `Thread` and `Process`
//! structures are *transitional* implementations; they feed into these public
//! types through explicit bridge layers rather than being exposed directly.
//!
//! ## Transitional mapping (Phases 1–4, 7–8)
//!
//! | Canonical concept        | Current internal backing                  | Future direction              |
//! |--------------------------|-------------------------------------------|-------------------------------|
//! | `task::Task`             | kernel `Thread`                           | becomes the Task impl         |
//! | `job::Job`               | kernel `Process` (partial)                | hollowed out by further phases|
//! | `job::JobExit`           | `Thread::exit_code`                       | Phase 3 exit path             |
//! | `job::JobWaitResult`     | `poll_task_exit` result                   | Phase 3 wait path             |
//! | `group::Group`           | `Process::pgid` / `ConsoleTtyState`       | Phase 4 coordination domain   |
//! | `authority::Authority`   | `ProcessSnapshot::name` (transitional)    | Phase 7 permission context    |
//! | `place::Place`           | `Process::cwd` + `namespace` (Phase 8)   | Phase 8 world context         |
//!
//! Public truth changes first; internal machinery follows.

extern crate alloc;

pub mod authority;
pub mod group;
pub mod job;
pub mod place;
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
