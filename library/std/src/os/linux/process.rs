//! Linux-specific extensions to primitives in the [`std::process`] module.
//!
//! [`std::process`]: crate::process

#![unstable(feature = "linux_pidfd", issue = "82971")]

use crate::io::Result;
use crate::os::unix::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::process;
use crate::sealed::Sealed;
#[cfg(not(doc))]
use crate::sys::fd::FileDesc;
use crate::sys_common::{AsInner, AsInnerMut, FromInner, IntoInner};

#[cfg(doc)]
struct FileDesc;

/// This type represents a file descriptor that refers to a process.
///
/// A `PidFd` can be obtained by setting the corresponding option on [`Command`]
/// with [`create_pidfd`]. Subsequently, the created pidfd can be retrieved
/// from the [`Child`] by calling [`pidfd`] or [`take_pidfd`].
///
/// Example:
/// ```no_run
/// #![feature(linux_pidfd)]
/// use std::os::linux::process::{CommandExt, ChildExt};
/// use std::process::Command;
///
/// let mut child = Command::new("echo")
///     .create_pidfd(true)
///     .spawn()
///     .expect("Failed to spawn child");
///
/// let pidfd = child
///     .take_pidfd()
///     .expect("Failed to retrieve pidfd");
///
/// // The file descriptor will be closed when `pidfd` is dropped.
/// ```
/// Refer to the man page of [`pidfd_open(2)`] for further details.
///
/// [`Command`]: process::Command
/// [`create_pidfd`]: CommandExt::create_pidfd
/// [`Child`]: process::Child
/// [`pidfd`]: fn@ChildExt::pidfd
/// [`take_pidfd`]: ChildExt::take_pidfd
/// [`pidfd_open(2)`]: https://man7.org/linux/man-pages/man2/pidfd_open.2.html
#[derive(Debug)]
pub struct PidFd {
    inner: FileDesc,
}

impl AsInner<FileDesc> for PidFd {
    fn as_inner(&self) -> &FileDesc {
        &self.inner
    }
}

impl FromInner<FileDesc> for PidFd {
    fn from_inner(inner: FileDesc) -> PidFd {
        PidFd { inner }
    }
}

impl IntoInner<FileDesc> for PidFd {
    fn into_inner(self) -> FileDesc {
        self.inner
    }
}

impl AsRawFd for PidFd {
    fn as_raw_fd(&self) -> RawFd {
        self.as_inner().as_raw_fd()
    }
}

impl FromRawFd for PidFd {
    unsafe fn from_raw_fd(fd: RawFd) -> Self {
        Self::from_inner(FileDesc::from_raw_fd(fd))
    }
}

impl IntoRawFd for PidFd {
    fn into_raw_fd(self) -> RawFd {
        self.into_inner().into_raw_fd()
    }
}

impl AsFd for PidFd {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

impl From<OwnedFd> for PidFd {
    fn from(fd: OwnedFd) -> Self {
        Self::from_inner(FileDesc::from_inner(fd))
    }
}

impl From<PidFd> for OwnedFd {
    fn from(pid_fd: PidFd) -> Self {
        pid_fd.into_inner().into_inner()
    }
}

/// Os-specific extensions for [`Child`]
///
/// [`Child`]: process::Child
pub trait ChildExt: Sealed {
    /// Obtains a reference to the [`PidFd`] created for this [`Child`], if available.
    ///
    /// A pidfd will only be available if its creation was requested with
    /// [`create_pidfd`] when the corresponding [`Command`] was created.
    ///
    /// Even if requested, a pidfd may not be available due to an older
    /// version of Linux being in use, or if some other error occurred.
    ///
    /// [`Command`]: process::Command
    /// [`create_pidfd`]: CommandExt::create_pidfd
    /// [`Child`]: process::Child
    fn pidfd(&self) -> Result<&PidFd>;

    /// Takes ownership of the [`PidFd`] created for this [`Child`], if available.
    ///
    /// A pidfd will only be available if its creation was requested with
    /// [`create_pidfd`] when the corresponding [`Command`] was created.
    ///
    /// Even if requested, a pidfd may not be available due to an older
    /// version of Linux being in use, or if some other error occurred.
    ///
    /// [`Command`]: process::Command
    /// [`create_pidfd`]: CommandExt::create_pidfd
    /// [`Child`]: process::Child
    fn take_pidfd(&mut self) -> Result<PidFd>;
}

/// Os-specific extensions for [`Command`]
///
/// [`Command`]: process::Command
pub trait CommandExt: Sealed {
    /// Sets whether a [`PidFd`](struct@PidFd) should be created for the [`Child`]
    /// spawned by this [`Command`].
    /// By default, no pidfd will be created.
    ///
    /// The pidfd can be retrieved from the child with [`pidfd`] or [`take_pidfd`].
    ///
    /// A pidfd will only be created if it is possible to do so
    /// in a guaranteed race-free manner (e.g. if the `clone3` system call
    /// is supported). Otherwise, [`pidfd`] will return an error.
    ///
    /// [`Command`]: process::Command
    /// [`Child`]: process::Child
    /// [`pidfd`]: fn@ChildExt::pidfd
    /// [`take_pidfd`]: ChildExt::take_pidfd
    fn create_pidfd(&mut self, val: bool) -> &mut process::Command;
}

impl CommandExt for process::Command {
    fn create_pidfd(&mut self, val: bool) -> &mut process::Command {
        self.as_inner_mut().create_pidfd(val);
        self
    }
}
