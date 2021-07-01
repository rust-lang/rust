//! Windows-specific extensions to general I/O primitives.
//!
//! Just like raw pointers, raw Windows handles and sockets point to resources
//! with dynamic lifetimes, and they can dangle if they outlive their resources
//! or be forged if they're created from invalid values.
//!
//! This module provides three types for representing raw handles and sockets
//! with different ownership properties: raw, borrowed, and owned, which are
//! analogous to types used for representing pointers:
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
//! `RawSocket` usage as unsafe. Libraries are encouraged to migrate, either by
//! adding `unsafe` to APIs that dereference `RawHandle` and `RawSocket`
//! values, or by using to `BorrowedHandle`, `BorrowedSocket`, `OwnedHandle`,
//! or `OwnedSocket`.
//!
//! Like references, `BorrowedHandle` and `BorrowedSocket` values are tied to a
//! lifetime, to ensure that they don't outlive the resource they point to.
//! These are safe to use. `BorrowedHandle` and `BorrowedSocket` values may be
//! used in APIs which provide safe access to any system call except for
//! `CloseHandle`, `closesocket`, or any other call that would end the
//! dynamic lifetime of the resource without ending the lifetime of the
//! handle or socket.
//!
//! Like boxes, `OwnedHandle` and `OwnedSocket` values conceptually own the
//! resource they point to, and free (close) it when they are dropped.
//!
//! [`BorrowedHandle<'a>`]: crate::os::windows::io::BorrowedHandle
//! [`BorrowedSocket<'a>`]: crate::os::windows::io::BorrowedSocket

#![stable(feature = "rust1", since = "1.0.0")]

mod handle;
mod raw;
mod socket;

#[unstable(feature = "io_safety", issue = "87074")]
pub use handle::*;
#[stable(feature = "rust1", since = "1.0.0")]
pub use raw::*;
#[unstable(feature = "io_safety", issue = "87074")]
pub use socket::*;
