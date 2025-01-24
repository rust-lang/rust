//! System bindings for the wasi preview 2 target.
//!
//! This is the next evolution of the original wasi target, and is intended to
//! replace that target over time.
//!
//! To begin with, this target mirrors the wasi target 1 to 1, but over
//! time this will change significantly.

#[path = "../wasi/args.rs"]
pub mod args;
#[path = "../wasi/env.rs"]
pub mod env;
#[path = "../wasi/fd.rs"]
pub mod fd;
#[path = "../wasi/fs.rs"]
pub mod fs;
#[allow(unused)]
#[path = "../wasm/atomics/futex.rs"]
pub mod futex;
#[path = "../wasi/io.rs"]
pub mod io;

pub mod net;
#[path = "../wasi/os.rs"]
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
#[path = "../wasi/stdio.rs"]
pub mod stdio;
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

use helpers::err2io;
pub use helpers::{abort_internal, decode_error_kind, is_interrupted};

mod cabi_realloc;
