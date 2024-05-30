//! This module implements "drop bombs" intended for use by command wrappers to ensure that the
//! constructed commands are *eventually* executed. This is exactly like `rustc_errors::Diag`
//! where we force every `Diag` to be consumed or we emit a bug, but we panic instead.
//!
//! This is inspired by <https://docs.rs/drop_bomb/latest/drop_bomb/>.

use std::borrow::Cow;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub(crate) struct DropBomb {
    msg: Cow<'static, str>,
    defused: bool,
}

impl DropBomb {
    /// Arm a [`DropBomb`]. If the value is dropped without being [`defused`][Self::defused], then
    /// it will panic.
    pub(crate) fn arm<S: Into<Cow<'static, str>>>(message: S) -> DropBomb {
        DropBomb { msg: message.into(), defused: false }
    }

    /// Defuse the [`DropBomb`]. This will prevent the drop bomb from panicking when dropped.
    pub(crate) fn defuse(&mut self) {
        self.defused = true;
    }
}

impl Drop for DropBomb {
    fn drop(&mut self) {
        if !self.defused && !std::thread::panicking() {
            panic!("{}", self.msg)
        }
    }
}
