#![allow(unsafe_op_in_unsafe_fn)]

/// The configure builtins provides runtime support compiler-builtin features
/// which require dynamic initialization to work as expected, e.g. aarch64
/// outline-atomics.
mod configure_builtins;

/// The PAL (platform abstraction layer) contains platform-specific abstractions
/// for implementing the features in the other submodules, e.g. UNIX file
/// descriptors.
mod pal;

mod alloc;
mod personality;

pub mod anonymous_pipe;
pub mod args;
pub mod backtrace;
pub mod cmath;
pub mod env;
pub mod env_consts;
pub mod exit_guard;
pub mod fd;
pub mod fs;
pub mod io;
pub mod net;
pub mod os_str;
pub mod path;
pub mod platform_version;
pub mod process;
pub mod random;
pub mod stdio;
pub mod sync;
pub mod thread;
pub mod thread_local;

// FIXME(117276): remove this, move feature implementations into individual
//                submodules.
pub use pal::*;
