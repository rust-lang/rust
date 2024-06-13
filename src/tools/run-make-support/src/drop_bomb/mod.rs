//! This module implements "drop bombs" intended for use by command wrappers to ensure that the
//! constructed commands are *eventually* executed. This is exactly like `rustc_errors::Diag` where
//! we force every `Diag` to be consumed or we emit a bug, but we panic instead.
//!
//! This is adapted from <https://docs.rs/drop_bomb/latest/drop_bomb/> and simplified for our
//! purposes.

use std::ffi::{OsStr, OsString};
use std::panic;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub(crate) struct DropBomb {
    command: OsString,
    defused: bool,
    armed_line: u32,
}

impl DropBomb {
    /// Arm a [`DropBomb`]. If the value is dropped without being [`defused`][Self::defused], then
    /// it will panic. It is expected that the command wrapper uses `#[track_caller]` to help
    /// propagate the caller info from rmake.rs.
    #[track_caller]
    pub(crate) fn arm<S: AsRef<OsStr>>(command: S) -> DropBomb {
        DropBomb {
            command: command.as_ref().into(),
            defused: false,
            armed_line: panic::Location::caller().line(),
        }
    }

    /// Defuse the [`DropBomb`]. This will prevent the drop bomb from panicking when dropped.
    pub(crate) fn defuse(&mut self) {
        self.defused = true;
    }
}

impl Drop for DropBomb {
    fn drop(&mut self) {
        if !self.defused && !std::thread::panicking() {
            panic!(
                "command constructed but not executed at line {}: `{}`",
                self.armed_line,
                self.command.to_string_lossy()
            )
        }
    }
}
