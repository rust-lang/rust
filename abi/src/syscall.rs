//! ThingOS Syscall Constants
//!
//!
pub mod asm;
pub mod conv;

mod numbers {
    include!("numbers.rs");
}
pub use numbers::*;

// pollfds entry, iovec, and other types remain here so they can be shared
// between the kernel, stem, and userspace without additional crate deps.

/// A scatter-gather I/O vector element, passed to [`SYS_FS_READV`] and [`SYS_FS_WRITEV`].
///
/// Layout matches POSIX `struct iovec` so that future libc ports can alias
/// this directly.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct IoVec {
    /// Base address of the buffer (as a raw pointer-width integer).
    pub base: usize,
    /// Length of the buffer in bytes.
    pub len: usize,
}

/// Entry in the `pollfds` array passed to [`SYS_FS_POLL`].
///
/// Layout mirrors POSIX `struct pollfd` so that future libc ports can
/// alias this directly.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct PollFd {
    /// File descriptor to watch.
    pub fd: i32,
    /// Events to wait for (input, using [`poll_flags`]).
    pub events: u16,
    /// Events that occurred (output, filled by the kernel).
    pub revents: u16,
}
