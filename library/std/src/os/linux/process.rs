//! Linux-specific extensions to primitives in the `std::process` module.

#![unstable(feature = "linux_pidfd", issue = "none")]

use crate::process;
use crate::sys_common::AsInnerMut;
use crate::io::Result;

/// Os-specific extensions to [`process::Child`]
///
/// [`process::Child`]: crate::process::Child
pub trait ChildExt {
    /// Obtains the pidfd created for this child process, if available.
    ///
    /// A pidfd will only ever be available if `create_pidfd(true)` was called
    /// when the corresponding `Command` was created.
    ///
    /// Even if `create_pidfd(true)` is called, a pidfd may not be available
    /// due to an older version of Linux being in use, or if
    /// some other error occured.
    ///
    /// See `man pidfd_open` for more details about pidfds.
    fn pidfd(&self) -> Result<i32>;
}

/// Os-specific extensions to [`process::Command`]
///
/// [`process::Command`]: crate::process::Command
pub trait CommandExt {
    /// Sets whether or this `Command` will attempt to create a pidfd
    /// for the child. If this method is never called, a pidfd will
    /// not be crated.
    ///
    /// The pidfd can be retrieved from the child via [`ChildExt::pidfd`]
    ///
    /// A pidfd will only be created if it is possible to do so
    /// in a guaranteed race-free manner (e.g. if the `clone3` system call is
    /// supported). Otherwise, [`ChildExit::pidfd`] will return an error.
    fn create_pidfd(&mut self, val: bool) -> &mut process::Command;
}

impl CommandExt for process::Command {
    fn create_pidfd(&mut self, val: bool) -> &mut process::Command {
        self.as_inner_mut().create_pidfd(val);
        self
    }
}
