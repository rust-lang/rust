//! Windows-specific extensions to general I/O primitives.
//!
//! Just like raw pointers, raw Windows handles and sockets point to resources
//! with dynamic lifetimes, and they can dangle if they outlive their resources
//! or be forged if they're created from invalid values.
//!
//! This module provides three types for representing raw handles and sockets
//! with different ownership properties: raw, borrowed, and owned, which are
//! analogous to types used for representing pointers. These types reflect concepts of [I/O
//! safety][io-safety] on Windows.
//!
//! | Type                   | Analogous to |
//! | ---------------------- | ------------ |
//! | [`RawHandle`]          | `*const _`   |
//! | [`RawSocket`]          | `*const _`   |
//! |                        |              |
//! | [`BorrowedHandle<'a>`] | `&'a _`      |
//! | [`BorrowedSocket<'a>`] | `&'a _`      |
//! |                        |              |
//! | [`OwnedHandle`]        | `Box<_>`     |
//! | [`OwnedSocket`]        | `Box<_>`     |
//!
//! Like raw pointers, `RawHandle` and `RawSocket` values are primitive values.
//! And in new code, they should be considered unsafe to do I/O on (analogous
//! to dereferencing them). Rust did not always provide this guidance, so
//! existing code in the Rust ecosystem often doesn't mark `RawHandle` and
//! `RawSocket` usage as unsafe.
//! Libraries are encouraged to migrate, either by adding `unsafe` to APIs
//! that dereference `RawHandle` and `RawSocket` values, or by using to
//! `BorrowedHandle`, `BorrowedSocket`, `OwnedHandle`, or `OwnedSocket`.
//!
//! Like references, `BorrowedHandle` and `BorrowedSocket` values are tied to a
//! lifetime, to ensure that they don't outlive the resource they point to.
//! These are safe to use. `BorrowedHandle` and `BorrowedSocket` values may be
//! used in APIs which provide safe access to any system call except for
//! `CloseHandle`, `closesocket`, or any other call that would end the
//! dynamic lifetime of the resource without ending the lifetime of the
//! handle or socket.
//!
//! `BorrowedHandle` and `BorrowedSocket` values may be used in APIs which
//! provide safe access to `DuplicateHandle` and `WSADuplicateSocketW` and
//! related functions, so types implementing `AsHandle`, `AsSocket`,
//! `From<OwnedHandle>`, or `From<OwnedSocket>` should not assume they always
//! have exclusive access to the underlying object.
//!
//! Like boxes, `OwnedHandle` and `OwnedSocket` values conceptually own the
//! resource they point to, and free (close) it when they are dropped.
//!
//! See the [`io` module docs][io-safety] for a general explanation of I/O safety.
//!
//! [`BorrowedHandle<'a>`]: crate::os::windows::io::BorrowedHandle
//! [`BorrowedSocket<'a>`]: crate::os::windows::io::BorrowedSocket
//! [io-safety]: crate::io#io-safety

#![stable(feature = "rust1", since = "1.0.0")]

mod handle;
mod raw;
mod socket;

#[stable(feature = "io_safety", since = "1.63.0")]
pub use handle::*;
#[stable(feature = "rust1", since = "1.0.0")]
pub use raw::*;
#[stable(feature = "io_safety", since = "1.63.0")]
pub use socket::*;

use crate::io::{self, Stderr, StderrLock, Stdin, StdinLock, Stdout, StdoutLock, Write};
use crate::ptr;
#[cfg(not(doc))]
use crate::sys::c;

#[cfg(test)]
mod tests;

#[unstable(feature = "stdio_swap", issue = "150667", reason = "recently added")]
pub impl(self) trait StdioExt {
    /// Sets the stdio console handle to `handle`, or `NULL` if it is `None`.
    /// The old handle, if any, will not be closed, i.e. it is leaked because
    /// console handles are shared global resources.
    ///
    /// Rust std::io write buffers (if any) are flushed, but other runtimes
    /// (e.g. C stdio) or libraries that acquire a clone of the file handle
    /// will not be aware of this change.
    ///
    /// ```
    /// #![feature(stdio_swap)]
    /// use std::io::{self, Read, Write};
    /// use std::os::windows::io::StdioExt;
    ///
    /// fn main() -> io::Result<()> {
    ///    let (reader, mut writer) = io::pipe()?;
    ///    let mut stdin = io::stdin();
    ///    stdin.set_handle(Some(reader))?;
    ///    writer.write_all(b"Hello, world!")?;
    ///    let mut buffer = vec![0; 13];
    ///    assert_eq!(stdin.read(&mut buffer)?, 13);
    ///    assert_eq!(&buffer, b"Hello, world!");
    ///    Ok(())
    /// }
    /// ```
    fn set_handle<T: Into<OwnedHandle>>(&mut self, handle: Option<T>) -> io::Result<()>;

    /// Sets the stdio console handle to `replace_with`. The previous handle is returned, or
    /// `None` if it was `NULL`.
    ///
    /// The returned handle is a `BorrowedHandle<'static>` because console handles are shared global resources
    /// and may have been obtained by other functions or threads.
    /// Only if you have ensured that no other part of the program has borrowed this handle you can convert it into
    /// an `OwnedHandle` and drop that to close it.
    ///
    /// Like `set_handle()`, Rust std::io write buffers (if any) are flushed.
    fn replace_handle<T: Into<OwnedHandle>>(
        &mut self,
        replace_with: T,
    ) -> io::Result<Option<BorrowedHandle<'static>>>;

    /// Sets the stdio console handle to `NULL` and returns the old one
    ///
    /// See [`set_handle()`] for additional details.
    ///
    /// [`set_handle()`]: StdioExt::set_handle
    fn take_handle(&mut self) -> io::Result<Option<BorrowedHandle<'static>>>;
}

macro io_ext_impl($stdio_ty:ty, $stdio_lock_ty:ty, $handle:path, $writer:literal) {
    #[unstable(feature = "stdio_swap", issue = "150667", reason = "recently added")]
    impl StdioExt for $stdio_ty {
        fn set_handle<T: Into<OwnedHandle>>(&mut self, handle: Option<T>) -> io::Result<()> {
            self.lock().set_handle(handle)
        }

        fn replace_handle<T: Into<OwnedHandle>>(
            &mut self,
            replace_with: T,
        ) -> io::Result<Option<BorrowedHandle<'static>>> {
            self.lock().replace_handle(replace_with)
        }

        fn take_handle(&mut self) -> io::Result<Option<BorrowedHandle<'static>>> {
            self.lock().take_handle()
        }
    }

    #[unstable(feature = "stdio_swap", issue = "150667", reason = "recently added")]
    impl StdioExt for $stdio_lock_ty {
        fn set_handle<T: Into<OwnedHandle>>(&mut self, handle: Option<T>) -> io::Result<()> {
            #[cfg($writer)]
            self.flush()?;
            let raw = handle.map(|h| h.into().into_raw_handle()).unwrap_or(ptr::null_mut());
            unsafe { c::SetStdHandle($handle, raw) };
            Ok(())
        }

        fn replace_handle<T: Into<OwnedHandle>>(
            &mut self,
            replace_with: T,
        ) -> io::Result<Option<BorrowedHandle<'static>>> {
            let old = unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) };
            self.set_handle(Some(replace_with))?;
            let handle = if old.as_raw_handle().is_null() { None } else { Some(old) };
            Ok(handle)
        }

        fn take_handle(&mut self) -> io::Result<Option<BorrowedHandle<'static>>> {
            let old = unsafe { BorrowedHandle::borrow_raw(self.as_raw_handle()) };
            #[cfg($writer)]
            self.flush()?;
            unsafe { c::SetStdHandle($handle, ptr::null_mut()) };
            let handle = if old.as_raw_handle().is_null() { None } else { Some(old) };
            Ok(handle)
        }
    }
}

io_ext_impl!(Stdout, StdoutLock<'_>, c::STD_OUTPUT_HANDLE, true);
io_ext_impl!(Stdin, StdinLock<'_>, c::STD_INPUT_HANDLE, false);
io_ext_impl!(Stderr, StderrLock<'_>, c::STD_ERROR_HANDLE, true);
