//! System bindings for the Trusty OS.

#[path = "../unsupported/args.rs"]
pub mod args;
#[path = "../unsupported/common.rs"]
#[deny(unsafe_op_in_unsafe_fn)]
mod common;
#[path = "../unsupported/env.rs"]
pub mod env;
#[path = "../unsupported/os.rs"]
pub mod os;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/thread.rs"]
pub mod thread;
#[path = "../unsupported/time.rs"]
pub mod time;

pub use common::*;
