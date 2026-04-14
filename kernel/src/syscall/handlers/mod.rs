//! Syscall handler implementations
//!
//! Organized into focused modules by function category.

mod device;
mod futex;
mod logging;
mod memory;
mod message;
mod port;
mod process;
mod random;
pub mod signal;
pub mod socket;
mod stdio;
mod time;
mod trace;
pub mod vfs;
mod wait;

// Re-export all syscall handlers
pub use device::*;
pub use futex::*;
pub use logging::*;
pub use memory::*;
pub use message::*;
pub use port::*;
pub use process::*;
pub use random::*;
pub use signal::*;
pub use socket::*;
pub use stdio::*;
pub use time::*;
pub use trace::*;
pub use wait::*;

// Shared utilities used by multiple handlers
use crate::syscall::validate::{copyin, copyout};
use abi::errors::{Errno, SysResult};
use alloc::string::String;
use core::sync::atomic::Ordering;

/// Blocking call to Root service (REMOVED)
pub(crate) fn root_call(_op: usize) -> SysResult<usize> {
    Err(Errno::ENOSYS)
}
