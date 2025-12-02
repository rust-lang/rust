//! Native threads.
//!
//! ## The threading model
//!
//! An executing Rust program consists of a collection of native OS threads,
//! each with their own stack and local state. Threads can be named, and
//! provide some built-in support for low-level synchronization.
//!
//! Communication between threads can be done through
//! [channels], Rust's message-passing types, along with [other forms of thread
//! synchronization](../../std/sync/index.html) and shared-memory data
//! structures. In particular, types that are guaranteed to be
//! threadsafe are easily shared between threads using the
//! atomically-reference-counted container, [`Arc`].
//!
//! Fatal logic errors in Rust cause *thread panic*, during which
//! a thread will unwind the stack, running destructors and freeing
//! owned resources. While not meant as a 'try/catch' mechanism, panics
//! in Rust can nonetheless be caught (unless compiling with `panic=abort`) with
//! [`catch_unwind`](../../std/panic/fn.catch_unwind.html) and recovered
//! from, or alternatively be resumed with
//! [`resume_unwind`](../../std/panic/fn.resume_unwind.html). If the panic
//! is not caught the thread will exit, but the panic may optionally be
//! detected from a different thread with [`join`]. If the main thread panics
//! without the panic being caught, the application will exit with a
//! non-zero exit code.
//!
//! When the main thread of a Rust program terminates, the entire program shuts
//! down, even if other threads are still running. However, this module provides
//! convenient facilities for automatically waiting for the termination of a
//! thread (i.e., join).
//!
//! ## Spawning a thread
//!
//! A new thread can be spawned using the [`thread::spawn`][`spawn`] function:
//!
//! ```rust
//! use std::thread;
//!
//! thread::spawn(move || {
//!     // some work here
//! });
//! ```
//!
//! In this example, the spawned thread is "detached," which means that there is
//! no way for the program to learn when the spawned thread completes or otherwise
//! terminates.
//!
//! To learn when a thread completes, it is necessary to capture the [`JoinHandle`]
//! object that is returned by the call to [`spawn`], which provides
//! a `join` method that allows the caller to wait for the completion of the
//! spawned thread:
//!
//! ```rust
//! use std::thread;
//!
//! let thread_join_handle = thread::spawn(move || {
//!     // some work here
//! });
//! // some work here
//! let res = thread_join_handle.join();
//! ```
//!
//! The [`join`] method returns a [`thread::Result`] containing [`Ok`] of the final
//! value produced by the spawned thread, or [`Err`] of the value given to
//! a call to [`panic!`] if the thread panicked.
//!
//! Note that there is no parent/child relationship between a thread that spawns a
//! new thread and the thread being spawned.  In particular, the spawned thread may or
//! may not outlive the spawning thread, unless the spawning thread is the main thread.
//!
//! ## Configuring threads
//!
//! A new thread can be configured before it is spawned via the [`Builder`] type,
//! which currently allows you to set the name and stack size for the thread:
//!
//! ```rust
//! # #![allow(unused_must_use)]
//! use std::thread;
//!
//! thread::Builder::new().name("thread1".to_string()).spawn(move || {
//!     println!("Hello, world!");
//! });
//! ```
//!
//! ## The `Thread` type
//!
//! Threads are represented via the [`Thread`] type, which you can get in one of
//! two ways:
//!
//! * By spawning a new thread, e.g., using the [`thread::spawn`][`spawn`]
//!   function, and calling [`thread`][`JoinHandle::thread`] on the [`JoinHandle`].
//! * By requesting the current thread, using the [`thread::current`] function.
//!
//! The [`thread::current`] function is available even for threads not spawned
//! by the APIs of this module.
//!
//! ## Thread-local storage
//!
//! This module also provides an implementation of thread-local storage for Rust
//! programs. Thread-local storage is a method of storing data into a global
//! variable that each thread in the program will have its own copy of.
//! Threads do not share this data, so accesses do not need to be synchronized.
//!
//! A thread-local key owns the value it contains and will destroy the value when the
//! thread exits. It is created with the [`thread_local!`] macro and can contain any
//! value that is `'static` (no borrowed pointers). It provides an accessor function,
//! [`with`], that yields a shared reference to the value to the specified
//! closure. Thread-local keys allow only shared access to values, as there would be no
//! way to guarantee uniqueness if mutable borrows were allowed. Most values
//! will want to make use of some form of **interior mutability** through the
//! [`Cell`] or [`RefCell`] types.
//!
//! ## Naming threads
//!
//! Threads are able to have associated names for identification purposes. By default, spawned
//! threads are unnamed. To specify a name for a thread, build the thread with [`Builder`] and pass
//! the desired thread name to [`Builder::name`]. To retrieve the thread name from within the
//! thread, use [`Thread::name`]. A couple of examples where the name of a thread gets used:
//!
//! * If a panic occurs in a named thread, the thread name will be printed in the panic message.
//! * The thread name is provided to the OS where applicable (e.g., `pthread_setname_np` in
//!   unix-like platforms).
//!
//! ## Stack size
//!
//! The default stack size is platform-dependent and subject to change.
//! Currently, it is 2 MiB on all Tier-1 platforms.
//!
//! There are two ways to manually specify the stack size for spawned threads:
//!
//! * Build the thread with [`Builder`] and pass the desired stack size to [`Builder::stack_size`].
//! * Set the `RUST_MIN_STACK` environment variable to an integer representing the desired stack
//!   size (in bytes). Note that setting [`Builder::stack_size`] will override this. Be aware that
//!   changes to `RUST_MIN_STACK` may be ignored after program start.
//!
//! Note that the stack size of the main thread is *not* determined by Rust.
//!
//! [channels]: crate::sync::mpsc
//! [`Arc`]: crate::sync::Arc
//! [`join`]: JoinHandle::join
//! [`Result`]: crate::result::Result
//! [`Ok`]: crate::result::Result::Ok
//! [`Err`]: crate::result::Result::Err
//! [`thread::current`]: current::current
//! [`thread::Result`]: Result
//! [`unpark`]: Thread::unpark
//! [`thread::park_timeout`]: park_timeout
//! [`Cell`]: crate::cell::Cell
//! [`RefCell`]: crate::cell::RefCell
//! [`with`]: LocalKey::with
//! [`thread_local!`]: crate::thread_local

#![stable(feature = "rust1", since = "1.0.0")]
#![deny(unsafe_op_in_unsafe_fn)]
// Under `test`, `__FastLocalKeyInner` seems unused.
#![cfg_attr(test, allow(dead_code))]

use crate::any::Any;

#[macro_use]
mod local;
mod builder;
mod current;
mod functions;
mod id;
mod join_handle;
mod lifecycle;
mod scoped;
mod spawnhook;
mod thread;

pub(crate) mod main_thread;

#[cfg(all(test, not(any(target_os = "emscripten", target_os = "wasi"))))]
mod tests;

#[stable(feature = "rust1", since = "1.0.0")]
pub use builder::Builder;
#[stable(feature = "rust1", since = "1.0.0")]
pub use current::current;
#[unstable(feature = "current_thread_id", issue = "147194")]
pub use current::current_id;
pub(crate) use current::{current_or_unnamed, current_os_id, drop_current, with_current_name};
#[stable(feature = "available_parallelism", since = "1.59.0")]
pub use functions::available_parallelism;
#[stable(feature = "park_timeout", since = "1.4.0")]
pub use functions::park_timeout;
#[stable(feature = "thread_sleep", since = "1.4.0")]
pub use functions::sleep;
#[unstable(feature = "thread_sleep_until", issue = "113752")]
pub use functions::sleep_until;
#[expect(deprecated)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use functions::{panicking, park, park_timeout_ms, sleep_ms, spawn, yield_now};
#[stable(feature = "thread_id", since = "1.19.0")]
pub use id::ThreadId;
#[stable(feature = "rust1", since = "1.0.0")]
pub use join_handle::JoinHandle;
pub(crate) use lifecycle::ThreadInit;
#[stable(feature = "rust1", since = "1.0.0")]
pub use local::{AccessError, LocalKey};
#[stable(feature = "scoped_threads", since = "1.63.0")]
pub use scoped::{Scope, ScopedJoinHandle, scope};
#[unstable(feature = "thread_spawn_hook", issue = "132951")]
pub use spawnhook::add_spawn_hook;
#[stable(feature = "rust1", since = "1.0.0")]
pub use thread::Thread;

// Implementation details used by the thread_local!{} macro.
#[doc(hidden)]
#[unstable(feature = "thread_local_internals", issue = "none")]
pub mod local_impl {
    pub use super::local::thread_local_process_attrs;
    pub use crate::sys::thread_local::*;
}

/// A specialized [`Result`] type for threads.
///
/// Indicates the manner in which a thread exited.
///
/// The value contained in the `Result::Err` variant
/// is the value the thread panicked with;
/// that is, the argument the `panic!` macro was called with.
/// Unlike with normal errors, this value doesn't implement
/// the [`Error`](crate::error::Error) trait.
///
/// Thus, a sensible way to handle a thread panic is to either:
///
/// 1. propagate the panic with [`std::panic::resume_unwind`]
/// 2. or in case the thread is intended to be a subsystem boundary
/// that is supposed to isolate system-level failures,
/// match on the `Err` variant and handle the panic in an appropriate way
///
/// A thread that completes without panicking is considered to exit successfully.
///
/// # Examples
///
/// Matching on the result of a joined thread:
///
/// ```no_run
/// use std::{fs, thread, panic};
///
/// fn copy_in_thread() -> thread::Result<()> {
///     thread::spawn(|| {
///         fs::copy("foo.txt", "bar.txt").unwrap();
///     }).join()
/// }
///
/// fn main() {
///     match copy_in_thread() {
///         Ok(_) => println!("copy succeeded"),
///         Err(e) => panic::resume_unwind(e),
///     }
/// }
/// ```
///
/// [`Result`]: crate::result::Result
/// [`std::panic::resume_unwind`]: crate::panic::resume_unwind
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(search_unbox)]
pub type Result<T> = crate::result::Result<T, Box<dyn Any + Send + 'static>>;

fn _assert_sync_and_send() {
    fn _assert_both<T: Send + Sync>() {}
    _assert_both::<JoinHandle<()>>();
    _assert_both::<Thread>();
}
