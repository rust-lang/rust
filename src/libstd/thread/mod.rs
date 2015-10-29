// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Native threads
//!
//! ## The threading model
//!
//! An executing Rust program consists of a collection of native OS threads,
//! each with their own stack and local state.
//!
//! Communication between threads can be done through
//! [channels](../../std/sync/mpsc/index.html), Rust's message-passing
//! types, along with [other forms of thread
//! synchronization](../../std/sync/index.html) and shared-memory data
//! structures. In particular, types that are guaranteed to be
//! threadsafe are easily shared between threads using the
//! atomically-reference-counted container,
//! [`Arc`](../../std/sync/struct.Arc.html).
//!
//! Fatal logic errors in Rust cause *thread panic*, during which
//! a thread will unwind the stack, running destructors and freeing
//! owned resources. Thread panic is unrecoverable from within
//! the panicking thread (i.e. there is no 'try/catch' in Rust), but
//! the panic may optionally be detected from a different thread. If
//! the main thread panics, the application will exit with a non-zero
//! exit code.
//!
//! When the main thread of a Rust program terminates, the entire program shuts
//! down, even if other threads are still running. However, this module provides
//! convenient facilities for automatically waiting for the termination of a
//! child thread (i.e., join).
//!
//! ## The `Thread` type
//!
//! Threads are represented via the `Thread` type, which you can
//! get in one of two ways:
//!
//! * By spawning a new thread, e.g. using the `thread::spawn` function.
//! * By requesting the current thread, using the `thread::current` function.
//!
//! Threads can be named, and provide some built-in support for low-level
//! synchronization (described below).
//!
//! The `thread::current()` function is available even for threads not spawned
//! by the APIs of this module.
//!
//! ## Spawning a thread
//!
//! A new thread can be spawned using the `thread::spawn` function:
//!
//! ```rust
//! use std::thread;
//!
//! thread::spawn(move || {
//!     // some work here
//! });
//! ```
//!
//! In this example, the spawned thread is "detached" from the current
//! thread. This means that it can outlive its parent (the thread that spawned
//! it), unless this parent is the main thread.
//!
//! The parent thread can also wait on the completion of the child
//! thread; a call to `spawn` produces a `JoinHandle`, which provides
//! a `join` method for waiting:
//!
//! ```rust
//! use std::thread;
//!
//! let child = thread::spawn(move || {
//!     // some work here
//! });
//! // some work here
//! let res = child.join();
//! ```
//!
//! The `join` method returns a `Result` containing `Ok` of the final
//! value produced by the child thread, or `Err` of the value given to
//! a call to `panic!` if the child panicked.
//!
//! ## Configuring threads
//!
//! A new thread can be configured before it is spawned via the `Builder` type,
//! which currently allows you to set the name and stack size for the child thread:
//!
//! ```rust
//! # #![allow(unused_must_use)]
//! use std::thread;
//!
//! thread::Builder::new().name("child1".to_string()).spawn(move || {
//!     println!("Hello, world!");
//! });
//! ```
//!
//! ## Blocking support: park and unpark
//!
//! Every thread is equipped with some basic low-level blocking support, via the
//! `park` and `unpark` functions.
//!
//! Conceptually, each `Thread` handle has an associated token, which is
//! initially not present:
//!
//! * The `thread::park()` function blocks the current thread unless or until
//!   the token is available for its thread handle, at which point it atomically
//!   consumes the token. It may also return *spuriously*, without consuming the
//!   token. `thread::park_timeout()` does the same, but allows specifying a
//!   maximum time to block the thread for.
//!
//! * The `unpark()` method on a `Thread` atomically makes the token available
//!   if it wasn't already.
//!
//! In other words, each `Thread` acts a bit like a semaphore with initial count
//! 0, except that the semaphore is *saturating* (the count cannot go above 1),
//! and can return spuriously.
//!
//! The API is typically used by acquiring a handle to the current thread,
//! placing that handle in a shared data structure so that other threads can
//! find it, and then `park`ing. When some desired condition is met, another
//! thread calls `unpark` on the handle.
//!
//! The motivation for this design is twofold:
//!
//! * It avoids the need to allocate mutexes and condvars when building new
//!   synchronization primitives; the threads already provide basic blocking/signaling.
//!
//! * It can be implemented very efficiently on many platforms.
//!
//! ## Thread-local storage
//!
//! This module also provides an implementation of thread local storage for Rust
//! programs. Thread local storage is a method of storing data into a global
//! variable which each thread in the program will have its own copy of.
//! Threads do not share this data, so accesses do not need to be synchronized.
//!
//! At a high level, this module provides two variants of storage:
//!
//! * Owned thread-local storage. This is a type of thread local key which
//!   owns the value that it contains, and will destroy the value when the
//!   thread exits. This variant is created with the `thread_local!` macro and
//!   can contain any value which is `'static` (no borrowed pointers).
//!
//! * Scoped thread-local storage. This type of key is used to store a reference
//!   to a value into local storage temporarily for the scope of a function
//!   call. There are no restrictions on what types of values can be placed
//!   into this key.
//!
//! Both forms of thread local storage provide an accessor function, `with`,
//! which will yield a shared reference to the value to the specified
//! closure. Thread-local keys only allow shared access to values as there is no
//! way to guarantee uniqueness if a mutable borrow was allowed. Most values
//! will want to make use of some form of **interior mutability** through the
//! `Cell` or `RefCell` types.

#![stable(feature = "rust1", since = "1.0.0")]

use prelude::v1::*;

use any::Any;
use cell::UnsafeCell;
use fmt;
use io;
use sync::{Mutex, Condvar, Arc};
use sys::thread as imp;
use sys_common::thread_info;
use sys_common::unwind;
use sys_common::util;
use time::Duration;

////////////////////////////////////////////////////////////////////////////////
// Thread-local storage
////////////////////////////////////////////////////////////////////////////////

#[macro_use] mod local;
#[macro_use] mod scoped_tls;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::local::{LocalKey, LocalKeyState};

#[unstable(feature = "scoped_tls",
           reason = "scoped TLS has yet to have wide enough use to fully \
                     consider stabilizing its interface",
           issue = "27715")]
pub use self::scoped_tls::ScopedKey;

#[doc(hidden)] pub use self::local::__KeyInner as __LocalKeyInner;
#[doc(hidden)] pub use self::scoped_tls::__KeyInner as __ScopedKeyInner;

////////////////////////////////////////////////////////////////////////////////
// Builder
////////////////////////////////////////////////////////////////////////////////

/// Thread configuration. Provides detailed control over the properties
/// and behavior of new threads.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Builder {
    // A name for the thread-to-be, for identification in panic messages
    name: Option<String>,
    // The size of the stack for the spawned thread
    stack_size: Option<usize>,
}

impl Builder {
    /// Generates the base configuration for spawning a thread, from which
    /// configuration methods can be chained.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> Builder {
        Builder {
            name: None,
            stack_size: None,
        }
    }

    /// Names the thread-to-be. Currently the name is used for identification
    /// only in panic messages.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn name(mut self, name: String) -> Builder {
        self.name = Some(name);
        self
    }

    /// Sets the size of the stack for the new thread.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn stack_size(mut self, size: usize) -> Builder {
        self.stack_size = Some(size);
        self
    }

    /// Spawns a new thread, and returns a join handle for it.
    ///
    /// The child thread may outlive the parent (unless the parent thread
    /// is the main thread; the whole process is terminated when the main
    /// thread finishes). The join handle can be used to block on
    /// termination of the child thread, including recovering its panics.
    ///
    /// # Errors
    ///
    /// Unlike the `spawn` free function, this method yields an
    /// `io::Result` to capture any failure to create the thread at
    /// the OS level.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn spawn<F, T>(self, f: F) -> io::Result<JoinHandle<T>> where
        F: FnOnce() -> T, F: Send + 'static, T: Send + 'static
    {
        let Builder { name, stack_size } = self;

        let stack_size = stack_size.unwrap_or(util::min_stack());

        let mut my_thread = Thread::new(name);
        let mut their_thread = my_thread.clone();

        let my_packet : Arc<UnsafeCell<Option<Result<T>>>>
            = Arc::new(UnsafeCell::new(None));
        let their_packet = my_packet.clone();

        let main = move |id: u32| {
            if let Some(name) = their_thread.name() {
                imp::Thread::set_name(name);
            }
            their_thread.id = id;
            unsafe {
                thread_info::set(imp::guard::current(), their_thread);
                let mut output = None;
                let try_result = {
                    let ptr = &mut output;
                    unwind::try(move || *ptr = Some(f()))
                };
                *their_packet.get() = Some(try_result.map(|()| {
                    output.unwrap()
                }));
            }
        };

        let native = unsafe {
            try!(imp::Thread::new(stack_size, Box::new(main)))
        };
        my_thread.id = native.id();
        Ok(JoinHandle(JoinInner {
            native: Some(native),
            thread: my_thread,
            packet: Packet(my_packet),
        }))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Free functions
////////////////////////////////////////////////////////////////////////////////

/// Spawns a new thread, returning a `JoinHandle` for it.
///
/// The join handle will implicitly *detach* the child thread upon being
/// dropped. In this case, the child thread may outlive the parent (unless
/// the parent thread is the main thread; the whole process is terminated when
/// the main thread finishes.) Additionally, the join handle provides a `join`
/// method that can be used to join the child thread. If the child thread
/// panics, `join` will return an `Err` containing the argument given to
/// `panic`.
///
/// # Panics
///
/// Panics if the OS fails to create a thread; use `Builder::spawn`
/// to recover from such errors.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn spawn<F, T>(f: F) -> JoinHandle<T> where
    F: FnOnce() -> T, F: Send + 'static, T: Send + 'static
{
    Builder::new().spawn(f).unwrap()
}

/// Gets a handle to the thread that invokes it.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn current() -> Thread {
    thread_info::current_thread().expect("use of std::thread::current() is not \
                                          possible after the thread's local \
                                          data has been destroyed")
}

/// Cooperatively gives up a timeslice to the OS scheduler.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn yield_now() {
    imp::Thread::yield_now()
}

/// Determines whether the current thread is unwinding because of panic.
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn panicking() -> bool {
    unwind::panicking()
}

/// Invokes a closure, capturing the cause of panic if one occurs.
///
/// This function will return `Ok` with the closure's result if the closure
/// does not panic, and will return `Err(cause)` if the closure panics. The
/// `cause` returned is the object with which panic was originally invoked.
///
/// It is currently undefined behavior to unwind from Rust code into foreign
/// code, so this function is particularly useful when Rust is called from
/// another language (normally C). This can run arbitrary Rust code, capturing a
/// panic and allowing a graceful handling of the error.
///
/// It is **not** recommended to use this function for a general try/catch
/// mechanism. The `Result` type is more appropriate to use for functions that
/// can fail on a regular basis.
///
/// The closure provided is required to adhere to the `'static` bound to ensure
/// that it cannot reference data in the parent stack frame, mitigating problems
/// with exception safety. Furthermore, a `Send` bound is also required,
/// providing the same safety guarantees as `thread::spawn` (ensuring the
/// closure is properly isolated from the parent).
///
/// # Examples
///
/// ```
/// #![feature(catch_panic)]
///
/// use std::thread;
///
/// let result = thread::catch_panic(|| {
///     println!("hello!");
/// });
/// assert!(result.is_ok());
///
/// let result = thread::catch_panic(|| {
///     panic!("oh no!");
/// });
/// assert!(result.is_err());
/// ```
#[unstable(feature = "catch_panic", reason = "recent API addition",
           issue = "27719")]
pub fn catch_panic<F, R>(f: F) -> Result<R>
    where F: FnOnce() -> R + Send + 'static
{
    let mut result = None;
    unsafe {
        let result = &mut result;
        try!(unwind::try(move || *result = Some(f())))
    }
    Ok(result.unwrap())
}

/// Puts the current thread to sleep for the specified amount of time.
///
/// The thread may sleep longer than the duration specified due to scheduling
/// specifics or platform-dependent functionality. Note that on unix platforms
/// this function will not return early due to a signal being received or a
/// spurious wakeup.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn sleep_ms(ms: u32) {
    sleep(Duration::from_millis(ms as u64))
}

/// Puts the current thread to sleep for the specified amount of time.
///
/// The thread may sleep longer than the duration specified due to scheduling
/// specifics or platform-dependent functionality.
///
/// # Platform behavior
///
/// On Unix platforms this function will not return early due to a
/// signal being received or a spurious wakeup. Platforms which do not support
/// nanosecond precision for sleeping will have `dur` rounded up to the nearest
/// granularity of time they can sleep for.
#[stable(feature = "thread_sleep", since = "1.4.0")]
pub fn sleep(dur: Duration) {
    imp::Thread::sleep(dur)
}

/// Blocks unless or until the current thread's token is made available.
///
/// Every thread is equipped with some basic low-level blocking support, via
/// the `park()` function and the [`unpark()`][unpark] method. These can be
/// used as a more CPU-efficient implementation of a spinlock.
///
/// [unpark]: struct.Thread.html#method.unpark
///
/// The API is typically used by acquiring a handle to the current thread,
/// placing that handle in a shared data structure so that other threads can
/// find it, and then parking (in a loop with a check for the token actually
/// being acquired).
///
/// A call to `park` does not guarantee that the thread will remain parked
/// forever, and callers should be prepared for this possibility.
///
/// See the [module documentation][thread] for more detail.
///
/// [thread]: index.html
//
// The implementation currently uses the trivial strategy of a Mutex+Condvar
// with wakeup flag, which does not actually allow spurious wakeups. In the
// future, this will be implemented in a more efficient way, perhaps along the lines of
//   http://cr.openjdk.java.net/~stefank/6989984.1/raw_files/new/src/os/linux/vm/os_linux.cpp
// or futuxes, and in either case may allow spurious wakeups.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn park() {
    let thread = current();
    let mut guard = thread.inner.lock.lock().unwrap();
    while !*guard {
        guard = thread.inner.cvar.wait(guard).unwrap();
    }
    *guard = false;
}

/// Blocks unless or until the current thread's token is made available or
/// the specified duration has been reached (may wake spuriously).
///
/// The semantics of this function are equivalent to `park()` except that the
/// thread will be blocked for roughly no longer than *ms*. This method
/// should not be used for precise timing due to anomalies such as
/// preemption or platform differences that may not cause the maximum
/// amount of time waited to be precisely *ms* long.
///
/// See the module doc for more detail.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn park_timeout_ms(ms: u32) {
    park_timeout(Duration::from_millis(ms as u64))
}

/// Blocks unless or until the current thread's token is made available or
/// the specified duration has been reached (may wake spuriously).
///
/// The semantics of this function are equivalent to `park()` except that the
/// thread will be blocked for roughly no longer than *dur*. This method
/// should not be used for precise timing due to anomalies such as
/// preemption or platform differences that may not cause the maximum
/// amount of time waited to be precisely *dur* long.
///
/// See the module doc for more detail.
///
/// # Platform behavior
///
/// Platforms which do not support nanosecond precision for sleeping will have
/// `dur` rounded up to the nearest granularity of time they can sleep for.
#[stable(feature = "park_timeout", since = "1.4.0")]
pub fn park_timeout(dur: Duration) {
    let thread = current();
    let mut guard = thread.inner.lock.lock().unwrap();
    if !*guard {
        let (g, _) = thread.inner.cvar.wait_timeout(guard, dur).unwrap();
        guard = g;
    }
    *guard = false;
}

////////////////////////////////////////////////////////////////////////////////
// Thread
////////////////////////////////////////////////////////////////////////////////

/// The internal representation of a `Thread` handle
struct Inner {
    name: Option<String>,
    lock: Mutex<bool>,          // true when there is a buffered unpark
    cvar: Condvar,
}

#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
/// A handle to a thread.
pub struct Thread {
    inner: Arc<Inner>,
    id: u32,
}

impl Thread {
    // Used only internally to construct a thread object without spawning
    fn new(name: Option<String>) -> Thread {
        Thread {
            inner: Arc::new(Inner {
                name: name,
                lock: Mutex::new(false),
                cvar: Condvar::new(),
            }),
            id: 0,
        }
    }

    /// Atomically makes the handle's token available if it is not already.
    ///
    /// See the module doc for more detail.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unpark(&self) {
        let mut guard = self.inner.lock.lock().unwrap();
        if !*guard {
            *guard = true;
            self.inner.cvar.notify_one();
        }
    }

    /// Gets the thread's name.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn name(&self) -> Option<&str> {
        self.inner.name.as_ref().map(|s| &**s)
    }

    /// Gets the thread's unique ID.
    ///
    /// This ID is unique across running threads, and a thread that has stopped
    /// might see its ID reused.
    pub fn id(&self) -> u32 {
        self.id
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Thread {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&self.name(), f)  // print ID?
    }
}

// a hack to get around privacy restrictions
impl thread_info::NewThread for Thread {
    fn new(name: Option<String>) -> Thread { Thread::new(name) }
}

////////////////////////////////////////////////////////////////////////////////
// JoinHandle
////////////////////////////////////////////////////////////////////////////////

/// Indicates the manner in which a thread exited.
///
/// A thread that completes without panicking is considered to exit successfully.
#[stable(feature = "rust1", since = "1.0.0")]
pub type Result<T> = ::result::Result<T, Box<Any + Send + 'static>>;

// This packet is used to communicate the return value between the child thread
// and the parent thread. Memory is shared through the `Arc` within and there's
// no need for a mutex here because synchronization happens with `join()` (the
// parent thread never reads this packet until the child has exited).
//
// This packet itself is then stored into a `JoinInner` which in turns is placed
// in `JoinHandle` and `JoinGuard`. Due to the usage of `UnsafeCell` we need to
// manually worry about impls like Send and Sync. The type `T` should
// already always be Send (otherwise the thread could not have been created) and
// this type is inherently Sync because no methods take &self. Regardless,
// however, we add inheriting impls for Send/Sync to this type to ensure it's
// Send/Sync and that future modifications will still appropriately classify it.
struct Packet<T>(Arc<UnsafeCell<Option<Result<T>>>>);

unsafe impl<T: Send> Send for Packet<T> {}
unsafe impl<T: Sync> Sync for Packet<T> {}

/// Inner representation for JoinHandle
struct JoinInner<T> {
    native: Option<imp::Thread>,
    thread: Thread,
    packet: Packet<T>,
}

impl<T> JoinInner<T> {
    fn join(&mut self) -> Result<T> {
        self.native.take().unwrap().join();
        unsafe {
            (*self.packet.0.get()).take().unwrap()
        }
    }
}

/// An owned permission to join on a thread (block on its termination).
///
/// A `JoinHandle` *detaches* the child thread when it is dropped.
///
/// Due to platform restrictions, it is not possible to `Clone` this
/// handle: the ability to join a child thread is a uniquely-owned
/// permission.
#[stable(feature = "rust1", since = "1.0.0")]
pub struct JoinHandle<T>(JoinInner<T>);

impl<T> JoinHandle<T> {
    /// Extracts a handle to the underlying thread
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn thread(&self) -> &Thread {
        &self.0.thread
    }

    /// Waits for the associated thread to finish.
    ///
    /// If the child thread panics, `Err` is returned with the parameter given
    /// to `panic`.
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn join(mut self) -> Result<T> {
        self.0.join()
    }
}

fn _assert_sync_and_send() {
    fn _assert_both<T: Send + Sync>() {}
    _assert_both::<JoinHandle<()>>();
    _assert_both::<Thread>();
}

////////////////////////////////////////////////////////////////////////////////
// Tests
////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use prelude::v1::*;

    use any::Any;
    use sync::mpsc::{channel, Sender};
    use result;
    use super::{Builder};
    use thread;
    use time::Duration;
    use u32;

    // !!! These tests are dangerous. If something is buggy, they will hang, !!!
    // !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

    #[test]
    fn test_unnamed_thread() {
        thread::spawn(move|| {
            assert!(thread::current().name().is_none());
        }).join().ok().unwrap();
    }

    #[test]
    fn test_named_thread() {
        Builder::new().name("ada lovelace".to_string()).spawn(move|| {
            assert!(thread::current().name().unwrap() == "ada lovelace".to_string());
        }).unwrap().join().unwrap();
    }

    #[test]
    fn test_run_basic() {
        let (tx, rx) = channel();
        thread::spawn(move|| {
            tx.send(()).unwrap();
        });
        rx.recv().unwrap();
    }

    #[test]
    fn test_join_panic() {
        match thread::spawn(move|| {
            panic!()
        }).join() {
            result::Result::Err(_) => (),
            result::Result::Ok(()) => panic!()
        }
    }

    #[test]
    fn test_spawn_sched() {
        use clone::Clone;

        let (tx, rx) = channel();

        fn f(i: i32, tx: Sender<()>) {
            let tx = tx.clone();
            thread::spawn(move|| {
                if i == 0 {
                    tx.send(()).unwrap();
                } else {
                    f(i - 1, tx);
                }
            });

        }
        f(10, tx);
        rx.recv().unwrap();
    }

    #[test]
    fn test_spawn_sched_childs_on_default_sched() {
        let (tx, rx) = channel();

        thread::spawn(move|| {
            thread::spawn(move|| {
                tx.send(()).unwrap();
            });
        });

        rx.recv().unwrap();
    }

    fn avoid_copying_the_body<F>(spawnfn: F) where F: FnOnce(Box<Fn() + Send>) {
        let (tx, rx) = channel();

        let x: Box<_> = box 1;
        let x_in_parent = (&*x) as *const i32 as usize;

        spawnfn(Box::new(move|| {
            let x_in_child = (&*x) as *const i32 as usize;
            tx.send(x_in_child).unwrap();
        }));

        let x_in_child = rx.recv().unwrap();
        assert_eq!(x_in_parent, x_in_child);
    }

    #[test]
    fn test_avoid_copying_the_body_spawn() {
        avoid_copying_the_body(|v| {
            thread::spawn(move || v());
        });
    }

    #[test]
    fn test_avoid_copying_the_body_thread_spawn() {
        avoid_copying_the_body(|f| {
            thread::spawn(move|| {
                f();
            });
        })
    }

    #[test]
    fn test_avoid_copying_the_body_join() {
        avoid_copying_the_body(|f| {
            let _ = thread::spawn(move|| {
                f()
            }).join();
        })
    }

    #[test]
    fn test_child_doesnt_ref_parent() {
        // If the child refcounts the parent thread, this will stack overflow when
        // climbing the thread tree to dereference each ancestor. (See #1789)
        // (well, it would if the constant were 8000+ - I lowered it to be more
        // valgrind-friendly. try this at home, instead..!)
        const GENERATIONS: u32 = 16;
        fn child_no(x: u32) -> Box<Fn() + Send> {
            return Box::new(move|| {
                if x < GENERATIONS {
                    thread::spawn(move|| child_no(x+1)());
                }
            });
        }
        thread::spawn(|| child_no(0)());
    }

    #[test]
    fn test_simple_newsched_spawn() {
        thread::spawn(move || {});
    }

    #[test]
    fn test_try_panic_message_static_str() {
        match thread::spawn(move|| {
            panic!("static string");
        }).join() {
            Err(e) => {
                type T = &'static str;
                assert!(e.is::<T>());
                assert_eq!(*e.downcast::<T>().unwrap(), "static string");
            }
            Ok(()) => panic!()
        }
    }

    #[test]
    fn test_try_panic_message_owned_str() {
        match thread::spawn(move|| {
            panic!("owned string".to_string());
        }).join() {
            Err(e) => {
                type T = String;
                assert!(e.is::<T>());
                assert_eq!(*e.downcast::<T>().unwrap(), "owned string".to_string());
            }
            Ok(()) => panic!()
        }
    }

    #[test]
    fn test_try_panic_message_any() {
        match thread::spawn(move|| {
            panic!(box 413u16 as Box<Any + Send>);
        }).join() {
            Err(e) => {
                type T = Box<Any + Send>;
                assert!(e.is::<T>());
                let any = e.downcast::<T>().unwrap();
                assert!(any.is::<u16>());
                assert_eq!(*any.downcast::<u16>().unwrap(), 413);
            }
            Ok(()) => panic!()
        }
    }

    #[test]
    fn test_try_panic_message_unit_struct() {
        struct Juju;

        match thread::spawn(move|| {
            panic!(Juju)
        }).join() {
            Err(ref e) if e.is::<Juju>() => {}
            Err(_) | Ok(()) => panic!()
        }
    }

    #[test]
    fn test_park_timeout_unpark_before() {
        for _ in 0..10 {
            thread::current().unpark();
            thread::park_timeout_ms(u32::MAX);
        }
    }

    #[test]
    fn test_park_timeout_unpark_not_called() {
        for _ in 0..10 {
            thread::park_timeout_ms(10);
        }
    }

    #[test]
    fn test_park_timeout_unpark_called_other_thread() {
        for _ in 0..10 {
            let th = thread::current();

            let _guard = thread::spawn(move || {
                super::sleep_ms(50);
                th.unpark();
            });

            thread::park_timeout_ms(u32::MAX);
        }
    }

    #[test]
    fn sleep_ms_smoke() {
        thread::sleep_ms(2);
    }

    // NOTE: the corresponding test for stderr is in run-pass/thread-stderr, due
    // to the test harness apparently interfering with stderr configuration.
}
