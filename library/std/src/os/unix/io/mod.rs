//! Unix-specific extensions to general I/O primitives.
//!
//! Just like raw pointers, raw file descriptors point to resources with
//! dynamic lifetimes, and they can dangle if they outlive their resources
//! or be forged if they're created from invalid values.
//!
//! This module provides three types for representing file descriptors,
//! with different ownership properties: raw, borrowed, and owned, which are
//! analogous to types used for representing pointers:
//!
//! | Type               | Analogous to |
//! | ------------------ | ------------ |
//! | [`RawFd`]          | `*const _`   |
//! | [`BorrowedFd<'a>`] | `&'a _`      |
//! | [`OwnedFd`]        | `Box<_>`     |
//!
//! Like raw pointers, `RawFd` values are primitive values. And in new code,
//! they should be considered unsafe to do I/O on (analogous to dereferencing
//! them). Rust did not always provide this guidance, so existing code in the
//! Rust ecosystem often doesn't mark `RawFd` usage as unsafe. Libraries are
//! encouraged to migrate, either by adding `unsafe` to APIs that dereference
//! `RawFd` values, or by using to `BorrowedFd` or `OwnedFd` instead.
//!
//! Like references, `BorrowedFd` values are tied to a lifetime, to ensure
//! that they don't outlive the resource they point to. These are safe to
//! use. `BorrowedFd` values may be used in APIs which provide safe access to
//! any system call except for:
//!  - `close`, because that would end the dynamic lifetime of the resource
//!    without ending the lifetime of the file descriptor.
//!  - `dup2`/`dup3`, in the second argument, because this argument is
//!    closed and assigned a new resource, which may break the assumptions
//!    other code using that file descriptor.
//! This list doesn't include `mmap`, since `mmap` does do a proper borrow of
//! its file descriptor argument. That said, `mmap` is unsafe for other
//! reasons: it operates on raw pointers, and it can have undefined behavior if
//! the underlying storage is mutated. Mutations may come from other processes,
//! or from the same process if the API provides `BorrowedFd` access, since as
//! mentioned earlier, `BorrowedFd` values may be used in APIs which provide
//! safe access to any system call. Consequently, code using `mmap` and
//! presenting a safe API must take full responsibility for ensuring that safe
//! Rust code cannot evoke undefined behavior through it.
//!
//! Like boxes, `OwnedFd` values conceptually own the resource they point to,
//! and free (close) it when they are dropped.
//!
//! [`BorrowedFd<'a>`]: crate::os::unix::io::BorrowedFd

#![stable(feature = "rust1", since = "1.0.0")]

mod fd;
mod raw;

#[unstable(feature = "io_safety", issue = "87074")]
pub use fd::*;
#[stable(feature = "rust1", since = "1.0.0")]
pub use raw::*;
