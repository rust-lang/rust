//! Platform Abstraction Layer (PAL)
//!
//! This module defines the explicit contract between Thing-OS applications
//! and the underlying platform. All platform-specific functionality must
//! go through this layer.
//!
//! # Design Principles
//!
//! - **Explicit over implicit**: Platform capabilities are explicitly surfaced
//! - **Minimal and stable**: Only essential platform primitives are exposed
//! - **Replaceable**: Implementations can be swapped without breaking consumers
//! - **No std leakage**: This layer ensures `no_std` compliance
//!
//! # Platform Surface
//!
//! The PAL provides:
//! - `log`: Logging primitives
//! - `clock`: Time and monotonic clock access
//! - `abort`: Panic and abort behavior
//! - `alloc`: Memory allocator hooks
//! - `net`: Network device access

pub mod abort;
#[cfg(feature = "global-alloc")]
pub mod alloc;
pub mod clock;
pub mod log;
pub mod net;

/// Shared syscall numbers included directly to avoid dependency on the full
/// `abi` crate (and its transitive `serde` dependency).
mod numbers {
    include!("../../../abi/src/numbers.rs");
}
pub use numbers::*;
