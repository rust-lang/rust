//! System bindings for the wasm/web platform
//!
//! This module contains the facade (aka platform-specific) implementations of
//! OS level functionality for wasm.
//!
//! This is all super highly experimental and not actually intended for
//! wide/production use yet, it's still all in the experimental category. This
//! will likely change over time.
//!
//! Currently all functions here are basically stubs that immediately return
//! errors. The hope is that with a portability lint we can turn actually just
//! remove all this and just omit parts of the standard library if we're
//! compiling for wasm. That way it's a compile time error for something that's
//! guaranteed to be a runtime error!

#[allow(unused)]
#[path = "../wasm/atomics/futex.rs"]
pub mod futex;

pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
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

pub(crate) use helpers::{abort_internal, decode_error_kind, err2io, is_interrupted};
