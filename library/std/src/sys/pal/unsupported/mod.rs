#![deny(unsafe_op_in_unsafe_fn)]

pub mod alloc;
pub mod args;
pub mod env;
pub mod fs;
pub mod io;
pub mod locks;
pub mod net;
pub mod once;
pub mod os;
pub mod pipe;
pub mod process;
pub mod stdio;
pub mod thread;
#[cfg(target_thread_local)]
pub mod thread_local_dtor;
pub mod thread_local_key;
pub mod thread_parking;
pub mod time;

mod common;
pub use common::*;
