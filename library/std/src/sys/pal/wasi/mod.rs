//! System bindings for the WASI platforms.
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for WASI. Currently this includes both WASIp1 and
//! WASIp2.

#[allow(unused)]
#[path = "../wasm/atomics/futex.rs"]
pub mod futex;

pub mod os;
pub mod stack_overflow;
#[path = "../unix/time.rs"]
pub mod time;

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
#[allow(unused)]
mod common;

pub use common::*;

mod helpers;

// The following exports are listed individually to work around Rust's glob
// import conflict rules. If we glob export `helpers` and `common` together,
// then the compiler complains about conflicts.

pub(crate) use helpers::abort_internal;
#[cfg(target_env = "p1")]
pub(crate) use helpers::err2io;
#[cfg(not(target_env = "p1"))]
pub use os::IsMinusOne;
pub use os::{cvt, cvt_r};

#[cfg(not(target_env = "p1"))]
mod cabi_realloc;
