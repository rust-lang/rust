//! Unix-specific extensions to general I/O primitives.
//!
//! Just like raw pointers, raw file descriptors point to resources with
//! dynamic lifetimes, and they can dangle if they outlive their resources
//! or be forged if they're created from invalid values.
//!
//! This module provides three types for representing file descriptors,
//! with different ownership properties: raw, borrowed, and owned, which are
//! analogous to types used for representing pointers. These types reflect concepts of [I/O
//! safety][io-safety] on Unix.
//!
//! | Type               | Analogous to |
//! | ------------------ | ------------ |
//! | [`RawFd`]          | `*const _`   |
//! | [`BorrowedFd<'a>`] | `&'a Arc<_>` |
//! | [`OwnedFd`]        | `Arc<_>`     |
//!
//! Like raw pointers, `RawFd` values are primitive values. And in new code,
//! they should be considered unsafe to do I/O on (analogous to dereferencing
//! them). Rust did not always provide this guidance, so existing code in the
//! Rust ecosystem often doesn't mark `RawFd` usage as unsafe.
//! Libraries are encouraged to migrate,
//! either by adding `unsafe` to APIs that dereference `RawFd` values, or by
//! using to `BorrowedFd` or `OwnedFd` instead.
//!
//! The use of `Arc` for borrowed/owned file descriptors may be surprising. Unix file descriptors
//! are mere references to internal kernel objects called "open file descriptions", and the same
//! open file description can be referenced by multiple file descriptors (e.g. if `dup` is used).
//! State such as the offset within the file is shared among all file descriptors that refer to the
//! same open file description, and the kernel internally does reference-counting to only close the
//! underlying resource once all file descriptors referencing it are closed. That's why `Arc` (and
//! not `Box`) is the closest Rust analogy to an "owned" file descriptor.
//!
//! Like references, `BorrowedFd` values are tied to a lifetime, to ensure
//! that they don't outlive the resource they point to. These are safe to
//! use. `BorrowedFd` values may be used in APIs which provide safe access to
//! any system call except for:
//!
//!  - `close`, because that would end the dynamic lifetime of the resource
//!    without ending the lifetime of the file descriptor. (Equivalently:
//!    an `&Arc<_>` cannot be `drop`ed.)
//!
//!  - `dup2`/`dup3`, in the second argument, because this argument is
//!    closed and assigned a new resource, which may break the assumptions of
//!    other code using that file descriptor.
//!
//! `BorrowedFd` values may be used in APIs which provide safe access to `dup` system calls, so code
//! working with `OwnedFd` cannot assume to have exclusive access to the underlying open file
//! description. (Equivalently: `&Arc` may be used in APIs that provide safe access to `clone`, so
//! code working with an `Arc` cannot assume that the reference count is 1.)
//!
//! `BorrowedFd` values may also be used with `mmap`, since `mmap` uses the
//! provided file descriptor in a manner similar to `dup` and does not require
//! the `BorrowedFd` passed to it to live for the lifetime of the resulting
//! mapping. That said, `mmap` is unsafe for other reasons: it operates on raw
//! pointers, and it can have undefined behavior if the underlying storage is
//! mutated. Mutations may come from other processes, or from the same process
//! if the API provides `BorrowedFd` access, since as mentioned earlier,
//! `BorrowedFd` values may be used in APIs which provide safe access to any
//! system call. Consequently, code using `mmap` and presenting a safe API must
//! take full responsibility for ensuring that safe Rust code cannot evoke
//! undefined behavior through it.
//!
//! Like `Arc`, `OwnedFd` values conceptually own one reference to the resource they point to,
//! and decrement the reference count when they are dropped (by calling `close`).
//! When the reference count reaches 0, the underlying open file description will be freed
//! by the kernel.
//!
//! See the [`io` module docs][io-safety] for a general explanation of I/O safety.
//!
//! ## `/proc/self/mem` and similar OS features
//!
//! Some platforms have special files, such as `/proc/self/mem`, which
//! provide read and write access to the process's memory. Such reads
//! and writes happen outside the control of the Rust compiler, so they do not
//! uphold Rust's memory safety guarantees.
//!
//! This does not mean that all APIs that might allow `/proc/self/mem`
//! to be opened and read from or written must be `unsafe`. Rust's safety guarantees
//! only cover what the program itself can do, and not what entities outside
//! the program can do to it. `/proc/self/mem` is considered to be such an
//! external entity, along with `/proc/self/fd/*`, debugging interfaces, and people with physical
//! access to the hardware. This is true even in cases where the program is controlling the external
//! entity.
//!
//! If you desire to comprehensively prevent programs from reaching out and
//! causing external entities to reach back in and violate memory safety, it's
//! necessary to use *sandboxing*, which is outside the scope of `std`.
//!
//! [`BorrowedFd<'a>`]: crate::os::unix::io::BorrowedFd
//! [io-safety]: crate::io#io-safety

#![stable(feature = "rust1", since = "1.0.0")]

use crate::io::{self, Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock, Write};
#[stable(feature = "rust1", since = "1.0.0")]
pub use crate::os::fd::*;
#[allow(unused_imports)] // not used on all targets
use crate::sys::cvt;

// Tests for this module
#[cfg(test)]
mod tests;

#[unstable(feature = "stdio_swap", issue = "150667", reason = "recently added")]
pub trait StdioExt: crate::sealed::Sealed {
    /// Redirects the stdio file descriptor to point to the file description underpinning `fd`.
    ///
    /// Rust std::io write buffers (if any) are flushed, but other runtimes
    /// (e.g. C stdio) or libraries that acquire a clone of the file descriptor
    /// will not be aware of this change.
    ///
    /// # Platform-specific behavior
    ///
    /// This is [currently] implemented using
    ///
    /// - `fd_renumber` on wasip1
    /// - `dup2` on most unixes
    ///
    /// [currently]: crate::io#platform-specific-behavior
    ///
    /// ```
    /// #![feature(stdio_swap)]
    /// use std::io::{self, Read, Write};
    /// use std::os::unix::io::StdioExt;
    ///
    /// fn main() -> io::Result<()> {
    ///    let (reader, mut writer) = io::pipe()?;
    ///    let mut stdin = io::stdin();
    ///    stdin.set_fd(reader)?;
    ///    writer.write_all(b"Hello, world!")?;
    ///    let mut buffer = vec![0; 13];
    ///    assert_eq!(stdin.read(&mut buffer)?, 13);
    ///    assert_eq!(&buffer, b"Hello, world!");
    ///    Ok(())
    /// }
    /// ```
    fn set_fd<T: Into<OwnedFd>>(&mut self, fd: T) -> io::Result<()>;

    /// Redirects the stdio file descriptor and returns a new `OwnedFd`
    /// backed by the previous file description.
    ///
    /// See [`set_fd()`] for details.
    ///
    /// [`set_fd()`]: StdioExt::set_fd
    fn replace_fd<T: Into<OwnedFd>>(&mut self, replace_with: T) -> io::Result<OwnedFd>;

    /// Redirects the stdio file descriptor to the null device (`/dev/null`)
    /// and returns a new `OwnedFd` backed by the previous file description.
    ///
    /// Programs that communicate structured data via stdio can use this early in `main()` to
    /// extract the fds, treat them as other IO types (`File`, `UnixStream`, etc),
    /// apply custom buffering or avoid interference from stdio use later in the program.
    ///
    /// See [`set_fd()`] for additional details.
    ///
    /// [`set_fd()`]: StdioExt::set_fd
    fn take_fd(&mut self) -> io::Result<OwnedFd>;
}

macro io_ext_impl($stdio_ty:ty, $stdio_lock_ty:ty, $writer:literal) {
    #[unstable(feature = "stdio_swap", issue = "150667", reason = "recently added")]
    impl StdioExt for $stdio_ty {
        fn set_fd<T: Into<OwnedFd>>(&mut self, fd: T) -> io::Result<()> {
            self.lock().set_fd(fd)
        }

        fn take_fd(&mut self) -> io::Result<OwnedFd> {
            self.lock().take_fd()
        }

        fn replace_fd<T: Into<OwnedFd>>(&mut self, replace_with: T) -> io::Result<OwnedFd> {
            self.lock().replace_fd(replace_with)
        }
    }

    #[unstable(feature = "stdio_swap", issue = "150667", reason = "recently added")]
    impl StdioExt for $stdio_lock_ty {
        fn set_fd<T: Into<OwnedFd>>(&mut self, fd: T) -> io::Result<()> {
            #[cfg($writer)]
            self.flush()?;
            replace_stdio_fd(self.as_fd(), fd.into())
        }

        fn take_fd(&mut self) -> io::Result<OwnedFd> {
            let null = null_fd()?;
            let cloned = self.as_fd().try_clone_to_owned()?;
            self.set_fd(null)?;
            Ok(cloned)
        }

        fn replace_fd<T: Into<OwnedFd>>(&mut self, replace_with: T) -> io::Result<OwnedFd> {
            let cloned = self.as_fd().try_clone_to_owned()?;
            self.set_fd(replace_with)?;
            Ok(cloned)
        }
    }
}

io_ext_impl!(Stdout, StdoutLock<'_>, true);
io_ext_impl!(Stdin, StdinLock<'_>, false);
io_ext_impl!(Stderr, StderrLock<'_>, true);

fn null_fd() -> io::Result<OwnedFd> {
    let null_dev = crate::fs::OpenOptions::new().read(true).write(true).open("/dev/null")?;
    Ok(null_dev.into())
}

/// Replaces the underlying file descriptor with the one from `other`.
/// Does not set CLOEXEC.
fn replace_stdio_fd(this: BorrowedFd<'_>, other: OwnedFd) -> io::Result<()> {
    cfg_select! {
        all(target_os = "wasi", target_env = "p1") => {
            cvt(unsafe { libc::__wasilibc_fd_renumber(other.as_raw_fd(), this.as_raw_fd()) }).map(|_| ())
        }
        not(any(
            target_arch = "wasm32",
            target_os = "hermit",
            target_os = "trusty",
            target_os = "motor"
        )) => {
            cvt(unsafe {libc::dup2(other.as_raw_fd(), this.as_raw_fd())}).map(|_| ())
        }
        _ => {
            let _ = (this, other);
            Err(io::Error::UNSUPPORTED_PLATFORM)
        }
    }
}
