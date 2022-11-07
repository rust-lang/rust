#![deny(unsafe_op_in_unsafe_fn)]

pub mod alloc;
#[path = "../unsupported/args.rs"]
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
#[path = "../unsupported/env.rs"]
pub mod env;
#[path = "../unsupported/fs.rs"]
pub mod fs;
#[path = "../unsupported/io.rs"]
pub mod io;
pub mod locks;
#[path = "../unsupported/net.rs"]
pub mod net;
#[path = "../unsupported/once.rs"]
pub mod once;
pub mod os;
#[path = "../unix/os_str.rs"]
pub mod os_str;
#[path = "../unix/path.rs"]
pub mod path;
#[path = "../unsupported/pipe.rs"]
pub mod pipe;
#[path = "../unsupported/process.rs"]
pub mod process;
pub mod stdio;
pub mod thread;
pub mod thread_local_key;
#[path = "../unsupported/thread_parking.rs"]
pub mod thread_parking;
pub mod time;

#[path = "../unsupported/common.rs"]
mod common;
pub use common::*;
