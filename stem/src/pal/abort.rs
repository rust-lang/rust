//! Abort and panic platform abstraction.
//!
//! Provides controlled termination primitives for abnormal exit conditions.

use crate::syscall::{debug_write, exit};

/// Abort execution with an error code.
///
/// This terminates the current process immediately. The exit code is
/// propagated to the parent process or kernel.
#[inline]
pub fn abort(exit_code: i32) -> ! {
    exit(exit_code)
}

/// Write to debug output channel.
///
/// Used primarily by panic handlers to emit diagnostics before aborting.
/// The debug channel bypasses normal logging infrastructure.
#[inline]
pub fn debug_write_str(s: &str) {
    let _ = debug_write(s, s.len());
}
