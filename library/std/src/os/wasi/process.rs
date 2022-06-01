//! Wasi-specific extensions to primitives in the [`std::process`] module.
//!
//! [`std::process`]: crate::process

#![stable(feature = "rust1", since = "1.0.0")]
#![allow(dead_code, unused)]

use crate::os::wasi::io::{AsFd, AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use crate::process;

#[stable(feature = "process_extensions", since = "1.2.0")]
impl FromRawFd for process::Stdio {
    #[inline]
    unsafe fn from_raw_fd(fd: RawFd) -> process::Stdio {
        unimplemented!()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<OwnedFd> for process::Stdio {
    #[inline]
    fn from(fd: OwnedFd) -> process::Stdio {
        unimplemented!()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdin {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        unimplemented!()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStdout {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        unimplemented!()
    }
}

#[stable(feature = "process_extensions", since = "1.2.0")]
impl AsRawFd for process::ChildStderr {
    #[inline]
    fn as_raw_fd(&self) -> RawFd {
        unimplemented!()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStdin {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        unimplemented!()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStdout {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        unimplemented!()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawFd for process::ChildStderr {
    #[inline]
    fn into_raw_fd(self) -> RawFd {
        unimplemented!()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for crate::process::ChildStdin {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        unimplemented!()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<crate::process::ChildStdin> for OwnedFd {
    #[inline]
    fn from(child_stdin: crate::process::ChildStdin) -> OwnedFd {
        unimplemented!()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for crate::process::ChildStdout {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        unimplemented!()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<crate::process::ChildStdout> for OwnedFd {
    #[inline]
    fn from(child_stdout: crate::process::ChildStdout) -> OwnedFd {
        unimplemented!()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl AsFd for crate::process::ChildStderr {
    #[inline]
    fn as_fd(&self) -> BorrowedFd<'_> {
        unimplemented!()
    }
}

#[unstable(feature = "io_safety", issue = "87074")]
impl From<crate::process::ChildStderr> for OwnedFd {
    #[inline]
    fn from(child_stderr: crate::process::ChildStderr) -> OwnedFd {
        unimplemented!()
    }
}
