//! System bindings for the wasi preview 2 target.
//!
//! This is the next evolution of the original wasi target, and is intended to
//! replace that target over time.
//!
//! To begin with, this target mirrors the wasi target 1 to 1, but over
//! time this will change significantly.

#[allow(unused)]
#[path = "../wasm/atomics/futex.rs"]
pub mod futex;

#[path = "../wasi/os.rs"]
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../wasi/thread.rs"]
pub mod thread;
#[path = "../wasi/time.rs"]
pub mod time;

#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
#[allow(unused)]
mod common;

pub use common::*;

#[path = "../wasi/helpers.rs"]
mod helpers;

// The following exports are listed individually to work around Rust's glob
// import conflict rules. If we glob export `helpers` and `common` together,
// then the compiler complains about conflicts.

pub(crate) use helpers::{abort_internal, decode_error_kind, err2io, is_interrupted};

mod cabi_realloc;
