//! Canonical Graph Types Registry
//!
//! This module defines the strict, packed, pointer-free structs used in the Thing-OS Graph ABI.

pub mod system; // Legacy/Syscall types (u64 based for now, slowly migrating)
pub use system::*;

// Re-export Wire IDs
pub use crate::wire::{BlobId, KindId, PredicateId, SymbolId, ThingId};
// Re-export Adapter
pub use crate::ids::HandleId;

// Graph atoms removed

// Logic/System
pub mod log_event;
pub mod process;
pub mod task;
pub mod thread;

// Assets
pub mod asset;
pub mod font;
pub mod window;

// Time
pub mod instant;

// Exports
pub use asset::Asset;
pub use font::Font;
pub use instant::{Duration, Instant};
pub use log_event::LogEvent;
pub use process::Process;
pub use task::Task;
pub use thread::Thread;
pub use window::Window;
