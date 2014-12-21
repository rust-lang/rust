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
//! [channels](../../std/comm/index.html), Rust's message-passing
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
//! panic may optionally be detected from a different thread. If
//! the main thread panics the application will exit with a non-zero
//! exit code.
//!
//! When the main thread of a Rust program terminates, the entire program shuts
//! down, even if other threads are still running. However, this module provides
//! convenient facilities for automatically waiting for the termination of a
//! child thread (i.e., join), described below.
//!
//! ## The `Thread` type
//!
//! Already-running threads are represented via the `Thread` type, which you can
//! get in one of two ways:
//!
//! * By spawning a new thread, e.g. using the `Thread::spawn` constructor;
//! * By requesting the current thread, using the `Thread::current` function.
//!
//! Threads can be named, and provide some built-in support for low-level
//! synchronization described below.
//!
//! The `Thread::current()` function is available even for threads not spawned
//! by the APIs of this module.
//!
//! ## Spawning a thread
//!
//! A new thread can be spawned using the `Thread::spawn` function:
//!
//! ```rust
//! use std::thread::Thread;
//!
//! let guard = Thread::spawn(move || {
//!     println!("Hello, World!");
//!     // some computation here
//! });
//! let result = guard.join();
//! ```
//!
//! The `spawn` function doesn't return a `Thread` directly; instead, it returns
//! a *join guard* from which a `Thread` can be extracted. The join guard is an
//! RAII-style guard that will automatically join the child thread (block until
//! it terminates) when it is dropped. You can join the child thread in advance
//! by calling the `join` method on the guard, which will also return the result
//! produced by the thread.
//!
//! If you instead wish to *detach* the child thread, allowing it to outlive its
//! parent, you can use the `detach` method on the guard,
//!
//! A handle to the thread itself is available via the `thread` method on the
//! join guard.
//!
//! ## Configuring threads
//!
//! A new thread can be configured before it is spawned via the `Builder` type,
//! which currently allows you to set the name, stack size, and writers for
//! `println!` and `panic!` for the child thread:
//!
//! ```rust
//! use std::thread;
//!
//! thread::Builder::new().name("child1".to_string()).spawn(move || {
//!     println!("Hello, world!")
//! }).detach();
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
//! * The `Thread::park()` function blocks the current thread unless or until
//!   the token is available for its thread handle, at which point It atomically
//!   consumes the token. It may also return *spuriously*, without consuming the
//!   token.
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
//! * It can be implemented highly efficiently on many platforms.

use any::Any;
use borrow::IntoCow;
use boxed::Box;
use cell::RacyCell;
use clone::Clone;
use kinds::{Send, Sync};
use ops::{Drop, FnOnce};
use option::Option::{mod, Some, None};
use result::Result::{Err, Ok};
use sync::{Mutex, Condvar, Arc};
use str::Str;
use string::String;
use rt::{mod, unwind};
use io::{Writer, stdio};
use thunk::Thunk;

use sys::thread as imp;
use sys_common::{stack, thread_info};

/// Thread configuation. Provides detailed control over the properties
/// and behavior of new threads.
pub struct Builder {
    // A name for the thread-to-be, for identification in panic messages
    name: Option<String>,
    // The size of the stack for the spawned thread
    stack_size: Option<uint>,
    // Thread-local stdout
    stdout: Option<Box<Writer + Send>>,
    // Thread-local stderr
    stderr: Option<Box<Writer + Send>>,
}

impl Builder {
    /// Generate the base configuration for spawning a thread, from which
    /// configuration methods can be chained.
    pub fn new() -> Builder {
        Builder {
            name: None,
            stack_size: None,
            stdout: None,
            stderr: None,
        }
    }

    /// Name the thread-to-be. Currently the name is used for identification
    /// only in panic messages.
    pub fn name(mut self, name: String) -> Builder {
        self.name = Some(name);
        self
    }

    /// Deprecated: use `name` instead
    #[deprecated = "use name instead"]
    pub fn named<T: IntoCow<'static, String, str>>(self, name: T) -> Builder {
        self.name(name.into_cow().into_owned())
    }

    /// Set the size of the stack for the new thread.
    pub fn stack_size(mut self, size: uint) -> Builder {
        self.stack_size = Some(size);
        self
    }

    /// Redirect thread-local stdout.
    #[experimental = "Will likely go away after proc removal"]
    pub fn stdout(mut self, stdout: Box<Writer + Send>) -> Builder {
        self.stdout = Some(stdout);
        self
    }

    /// Redirect thread-local stderr.
    #[experimental = "Will likely go away after proc removal"]
    pub fn stderr(mut self, stderr: Box<Writer + Send>) -> Builder {
        self.stderr = Some(stderr);
        self
    }

    /// Spawn a new joinable thread, and return a JoinGuard guard for it.
    ///
    /// See `Thead::spawn` and the module doc for more details.
    pub fn spawn<T, F>(self, f: F) -> JoinGuard<T> where
        T: Send, F: FnOnce() -> T, F: Send
    {
        self.spawn_inner(Thunk::new(f))
    }

    fn spawn_inner<T: Send>(self, f: Thunk<(), T>) -> JoinGuard<T> {
        let my_packet = Arc::new(RacyCell::new(None));
        let their_packet = my_packet.clone();

        let Builder { name, stack_size, stdout, stderr } = self;

        let stack_size = stack_size.unwrap_or(rt::min_stack());
        let my_thread = Thread::new(name);
        let their_thread = my_thread.clone();

        // Spawning a new OS thread guarantees that __morestack will never get
        // triggered, but we must manually set up the actual stack bounds once
        // this function starts executing. This raises the lower limit by a bit
        // because by the time that this function is executing we've already
        // consumed at least a little bit of stack (we don't know the exact byte
        // address at which our stack started).
        let main = move |:| {
            let something_around_the_top_of_the_stack = 1;
            let addr = &something_around_the_top_of_the_stack as *const int;
            let my_stack_top = addr as uint;
            let my_stack_bottom = my_stack_top - stack_size + 1024;
            unsafe {
                stack::record_os_managed_stack_bounds(my_stack_bottom, my_stack_top);
            }
            thread_info::set(
                (my_stack_bottom, my_stack_top),
                unsafe { imp::guard::current() },
                their_thread
            );

            let mut output = None;
            let f: Thunk<(), T> = if stdout.is_some() || stderr.is_some() {
                Thunk::new(move |:| {
                    let _ = stdout.map(stdio::set_stdout);
                    let _ = stderr.map(stdio::set_stderr);
                    f.invoke(())
                })
            } else {
                f
            };

            let try_result = {
                let ptr = &mut output;

                // There are two primary reasons that general try/catch is
                // unsafe. The first is that we do not support nested
                // try/catch. The fact that this is happening in a newly-spawned
                // thread suffices. The second is that unwinding while unwinding
                // is not defined.  We take care of that by having an
                // 'unwinding' flag in the thread itself. For these reasons,
                // this unsafety should be ok.
                unsafe {
                    unwind::try(move || *ptr = Some(f.invoke(())))
                }
            };
            unsafe {
                *their_packet.get() = Some(match (output, try_result) {
                    (Some(data), Ok(_)) => Ok(data),
                    (None, Err(cause)) => Err(cause),
                    _ => unreachable!()
                });
            }
        };

        JoinGuard {
            native: unsafe { imp::create(stack_size, Thunk::new(main)) },
            joined: false,
            packet: my_packet,
            thread: my_thread,
        }
    }
}

struct Inner {
    name: Option<String>,
    lock: Mutex<bool>,          // true when there is a buffered unpark
    cvar: Condvar,
}

unsafe impl Sync for Inner {}

#[deriving(Clone)]
/// A handle to a thread.
pub struct Thread {
    inner: Arc<Inner>,
}

unsafe impl Sync for Thread {}

impl Thread {
    // Used only internally to construct a thread object without spawning
    fn new(name: Option<String>) -> Thread {
        Thread {
            inner: Arc::new(Inner {
                name: name,
                lock: Mutex::new(false),
                cvar: Condvar::new(),
            })
        }
    }

    /// Spawn a new joinable thread, returning a `JoinGuard` for it.
    ///
    /// The join guard can be used to explicitly join the child thead (via
    /// `join`), returning `Result<T>`, or it will implicitly join the child
    /// upon being dropped. To detach the child, allowing it to outlive the
    /// current thread, use `detach`.  See the module documentation for additional details.
    pub fn spawn<T, F>(f: F) -> JoinGuard<T> where
        T: Send, F: FnOnce() -> T, F: Send
    {
        Builder::new().spawn(f)
    }

    /// Gets a handle to the thread that invokes it.
    pub fn current() -> Thread {
        thread_info::current_thread()
    }

    /// Cooperatively give up a timeslice to the OS scheduler.
    pub fn yield_now() {
        unsafe { imp::yield_now() }
    }

    /// Determines whether the current thread is panicking.
    pub fn panicking() -> bool {
        unwind::panicking()
    }

    /// Block unless or until the current thread's token is made available (may wake spuriously).
    ///
    /// See the module doc for more detail.
    //
    // The implementation currently uses the trivial strategy of a Mutex+Condvar
    // with wakeup flag, which does not actually allow spurious wakeups. In the
    // future, this will be implemented in a more efficient way, perhaps along the lines of
    //   http://cr.openjdk.java.net/~stefank/6989984.1/raw_files/new/src/os/linux/vm/os_linux.cpp
    // or futuxes, and in either case may allow spurious wakeups.
    pub fn park() {
        let thread = Thread::current();
        let mut guard = thread.inner.lock.lock();
        while !*guard {
            thread.inner.cvar.wait(&guard);
        }
        *guard = false;
    }

    /// Atomically makes the handle's token available if it is not already.
    ///
    /// See the module doc for more detail.
    pub fn unpark(&self) {
        let mut guard = self.inner.lock.lock();
        if !*guard {
            *guard = true;
            self.inner.cvar.notify_one();
        }
    }

    /// Get the thread's name.
    pub fn name(&self) -> Option<&str> {
        self.inner.name.as_ref().map(|s| s.as_slice())
    }
}

// a hack to get around privacy restrictions
impl thread_info::NewThread for Thread {
    fn new(name: Option<String>) -> Thread { Thread::new(name) }
}

/// Indicates the manner in which a thread exited.
///
/// A thread that completes without panicking is considered to exit successfully.
pub type Result<T> = ::result::Result<T, Box<Any + Send>>;

#[must_use]
/// An RAII-style guard that will block until thread termination when dropped.
///
/// The type `T` is the return type for the thread's main function.
pub struct JoinGuard<T> {
    native: imp::rust_thread,
    thread: Thread,
    joined: bool,
    packet: Arc<RacyCell<Option<Result<T>>>>,
}

impl<T: Send> JoinGuard<T> {
    /// Extract a handle to the thread this guard will join on.
    pub fn thread(&self) -> &Thread {
        &self.thread
    }

    /// Wait for the associated thread to finish, returning the result of the thread's
    /// calculation.
    ///
    /// If the child thread panics, `Err` is returned with the parameter given
    /// to `panic`.
    pub fn join(mut self) -> Result<T> {
        assert!(!self.joined);
        unsafe { imp::join(self.native) };
        self.joined = true;
        unsafe {
            (*self.packet.get()).take().unwrap()
        }
    }

    /// Detaches the child thread, allowing it to outlive its parent.
    pub fn detach(mut self) {
        unsafe { imp::detach(self.native) };
        self.joined = true; // avoid joining in the destructor
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for JoinGuard<T> {
    fn drop(&mut self) {
        if !self.joined {
            unsafe { imp::join(self.native) };
        }
    }
}

#[cfg(test)]
mod test {
    use prelude::*;
    use any::{Any, AnyRefExt};
    use boxed::BoxAny;
    use result;
    use std::io::{ChanReader, ChanWriter};
    use thunk::Thunk;
    use super::{Thread, Builder};

    // !!! These tests are dangerous. If something is buggy, they will hang, !!!
    // !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

    #[test]
    fn test_unnamed_thread() {
        Thread::spawn(move|| {
            assert!(Thread::current().name().is_none());
        }).join().map_err(|_| ()).unwrap();
    }

    #[test]
    fn test_named_thread() {
        Builder::new().name("ada lovelace".to_string()).spawn(move|| {
            assert!(Thread::current().name().unwrap() == "ada lovelace".to_string());
        }).join().map_err(|_| ()).unwrap();
    }

    #[test]
    fn test_run_basic() {
        let (tx, rx) = channel();
        Thread::spawn(move|| {
            tx.send(());
        }).detach();
        rx.recv();
    }

    #[test]
    fn test_join_success() {
        match Thread::spawn(move|| -> String {
            "Success!".to_string()
        }).join().as_ref().map(|s| s.as_slice()) {
            result::Result::Ok("Success!") => (),
            _ => panic!()
        }
    }

    #[test]
    fn test_join_panic() {
        match Thread::spawn(move|| {
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

        fn f(i: int, tx: Sender<()>) {
            let tx = tx.clone();
            Thread::spawn(move|| {
                if i == 0 {
                    tx.send(());
                } else {
                    f(i - 1, tx);
                }
            }).detach();

        }
        f(10, tx);
        rx.recv();
    }

    #[test]
    fn test_spawn_sched_childs_on_default_sched() {
        let (tx, rx) = channel();

        Thread::spawn(move|| {
            Thread::spawn(move|| {
                tx.send(());
            }).detach();
        }).detach();

        rx.recv();
    }

    fn avoid_copying_the_body<F>(spawnfn: F) where F: FnOnce(Thunk) {
        let (tx, rx) = channel::<uint>();

        let x = box 1;
        let x_in_parent = (&*x) as *const int as uint;

        spawnfn(Thunk::new(move|| {
            let x_in_child = (&*x) as *const int as uint;
            tx.send(x_in_child);
        }));

        let x_in_child = rx.recv();
        assert_eq!(x_in_parent, x_in_child);
    }

    #[test]
    fn test_avoid_copying_the_body_spawn() {
        avoid_copying_the_body(|v| {
            Thread::spawn(move || v.invoke(())).detach();
        });
    }

    #[test]
    fn test_avoid_copying_the_body_thread_spawn() {
        avoid_copying_the_body(|f| {
            Thread::spawn(move|| {
                f.invoke(());
            }).detach();
        })
    }

    #[test]
    fn test_avoid_copying_the_body_join() {
        avoid_copying_the_body(|f| {
            let _ = Thread::spawn(move|| {
                f.invoke(())
            }).join();
        })
    }

    #[test]
    fn test_child_doesnt_ref_parent() {
        // If the child refcounts the parent task, this will stack overflow when
        // climbing the task tree to dereference each ancestor. (See #1789)
        // (well, it would if the constant were 8000+ - I lowered it to be more
        // valgrind-friendly. try this at home, instead..!)
        static GENERATIONS: uint = 16;
        fn child_no(x: uint) -> Thunk {
            return Thunk::new(move|| {
                if x < GENERATIONS {
                    Thread::spawn(move|| child_no(x+1).invoke(())).detach();
                }
            });
        }
        Thread::spawn(|| child_no(0).invoke(())).detach();
    }

    #[test]
    fn test_simple_newsched_spawn() {
        Thread::spawn(move || {}).detach();
    }

    #[test]
    fn test_try_panic_message_static_str() {
        match Thread::spawn(move|| {
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
        match Thread::spawn(move|| {
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
        match Thread::spawn(move|| {
            panic!(box 413u16 as Box<Any + Send>);
        }).join() {
            Err(e) => {
                type T = Box<Any + Send>;
                assert!(e.is::<T>());
                let any = e.downcast::<T>().unwrap();
                assert!(any.is::<u16>());
                assert_eq!(*any.downcast::<u16>().unwrap(), 413u16);
            }
            Ok(()) => panic!()
        }
    }

    #[test]
    fn test_try_panic_message_unit_struct() {
        struct Juju;

        match Thread::spawn(move|| {
            panic!(Juju)
        }).join() {
            Err(ref e) if e.is::<Juju>() => {}
            Err(_) | Ok(()) => panic!()
        }
    }

    #[test]
    fn test_stdout() {
        let (tx, rx) = channel();
        let mut reader = ChanReader::new(rx);
        let stdout = ChanWriter::new(tx);

        let r = Builder::new().stdout(box stdout as Box<Writer + Send>).spawn(move|| {
            print!("Hello, world!");
        }).join();
        assert!(r.is_ok());

        let output = reader.read_to_string().unwrap();
        assert_eq!(output, "Hello, world!".to_string());
    }

    // NOTE: the corresponding test for stderr is in run-pass/task-stderr, due
    // to the test harness apparently interfering with stderr configuration.
}
