#![allow(unsafe_op_in_unsafe_fn)]

/// The PAL (platform abstraction layer) contains platform-specific abstractions
/// for implementing the features in the other submodules, e.g. UNIX file
/// descriptors.
mod pal;

mod alloc;
mod personality;

pub mod anonymous_pipe;
pub mod backtrace;
pub mod cmath;
pub mod exit_guard;
pub mod fd;
pub mod fs;
pub mod io;
pub mod net;
pub mod os_str;
pub mod path;
pub mod process;
pub mod random;
pub mod stdio;
pub mod sync;
pub mod thread_local;

// FIXME(117276): remove this, move feature implementations into individual
//                submodules.
pub use pal::*;
