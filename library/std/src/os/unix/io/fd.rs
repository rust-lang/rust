//! Owned and borrowed file descriptors.

#![unstable(feature = "io_safety", issue = "87074")]

// Tests for this module
#[cfg(test)]
mod tests;

use crate::sys_common::{AsInner, IntoInner};

pub use super::super::super::super::fd::*;

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for crate::process::ChildStdin {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<crate::process::ChildStdin> for OwnedFd {
    #[inline]
    fn from(child_stdin: crate::process::ChildStdin) -> OwnedFd {
        child_stdin.into_inner().into_inner().into_inner()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for crate::process::ChildStdout {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<crate::process::ChildStdout> for OwnedFd {
    #[inline]
    fn from(child_stdout: crate::process::ChildStdout) -> OwnedFd {
        child_stdout.into_inner().into_inner().into_inner()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for crate::process::ChildStderr {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.as_inner().as_fd()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<crate::process::ChildStderr> for OwnedFd {
    #[inline]
    fn from(child_stderr: crate::process::ChildStderr) -> OwnedFd {
        child_stderr.into_inner().into_inner().into_inner()
    }
}
