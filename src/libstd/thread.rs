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
//! Threads generally have their memory *isolated* from each other by virtue of
//! Rust's owned types (which of course may only be owned by a single thread at
//! a time). Communication between threads can be done through
//! [channels](../../std/comm/index.html), Rust's message-passing types, along
//! with [other forms of thread synchronization](../../std/sync/index.html) and
//! shared-memory data structures. In particular, types that are guaranteed to
//! be threadsafe are easily shared between threads using the
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
//! There are a few different ways to spawn a new thread, depending on how it
//! should relate to the parent thread.
//!
//! ### Simple detached threads
//!
//! The simplest case just spawns a completely independent (detached) thread,
//! returning a new `Thread` handle to it:
//!
//! ```rust
//! use std::thread::Thread;
//!
//! Thread::spawn(proc() {
//!     println!("Hello, World!");
//! })
//! ```
//!
//! The spawned thread may outlive its parent.
//!
//! ### Joining
//!
//! Alternatively, the `with_join` constructor spawns a new thread and returns a
//! `JoinGuard` which can be used to wait until the child thread completes,
//! returning its result (or `Err` if the child thread panicked):
//!
//! ```rust
//! use std::thread::Thread;
//!
//! let guard = Thread::with_join(proc() { panic!() };
//! assert!(guard.join().is_err());
//! ```
//!
//! The guard works in RAII style, meaning that the child thread is
//! automatically joined when the guard is dropped. A handle to the thread
//! itself is available via the `thread` method on the guard.
//!
//! ### Configured threads
//!
//! Finally, a new thread can be configured independently of how it is
//! spawned. Configuration is available via the `Cfg` builder, which currently
//! allows you to set the name, stack size, and writers for `println!` and
//! `panic!` for the child thread:
//!
//! ```rust
//! use std::thread;
//!
//! thread::cfg().name("child1").spawn(proc() { println!("Hello, world!") });
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

use core::prelude::*;

use any::Any;
use borrow::IntoCow;
use boxed::Box;
use mem;
use sync::{Mutex, Condvar, Arc};
use string::String;
use rt::{mod, unwind};
use io::{Writer, stdio};

use sys::thread as imp;
use sys_common::{stack, thread_info};

/// Thread configuation. Provides detailed control over the properties
/// and behavior of new threads.
pub struct Cfg {
    // A name for the thread-to-be, for identification in panic messages
    name: Option<String>,
    // The size of the stack for the spawned thread
    stack_size: Option<uint>,
    // Thread-local stdout
    stdout: Option<Box<Writer + Send>>,
    // Thread-local stderr
    stderr: Option<Box<Writer + Send>>,
}

impl Cfg {
    /// Generate the base configuration for spawning a thread, from which
    /// configuration methods can be chained.
    pub fn new() -> Cfg {
        Cfg {
            name: None,
            stack_size: None,
            stdout: None,
            stderr: None,
        }
    }

    /// Name the thread-to-be. Currently the name is used for identification
    /// only in panic messages.
    pub fn name(mut self, name: String) -> Cfg {
        self.name = Some(name);
        self
    }

    /// Deprecated: use `name` instead
    #[deprecated = "use name instead"]
    pub fn named<T: IntoCow<'static, String, str>>(self, name: T) -> Cfg {
        self.name(name.into_cow().into_owned())
    }

    /// Set the size of the stack for the new thread.
    pub fn stack_size(mut self, size: uint) -> Cfg {
        self.stack_size = Some(size);
        self
    }

    /// Redirect thread-local stdout.
    #[experimental = "Will likely go away after proc removal"]
    pub fn stdout(mut self, stdout: Box<Writer + Send>) -> Cfg {
        self.stdout = Some(stdout);
        self
    }

    /// Redirect thread-local stderr.
    #[experimental = "Will likely go away after proc removal"]
    pub fn stderr(mut self, stderr: Box<Writer + Send>) -> Cfg {
        self.stderr = Some(stderr);
        self
    }

    fn core_spawn<T: Send>(self, f: proc():Send -> T, after: proc(Result<T>):Send)
                           -> (imp::rust_thread, Thread)
    {
        let Cfg { name, stack_size, stdout, stderr } = self;

        let stack_size = stack_size.unwrap_or(rt::min_stack());
        let my_thread = Thread::new(name);
        let their_thread = my_thread.clone();

        // Spawning a new OS thread guarantees that __morestack will never get
        // triggered, but we must manually set up the actual stack bounds once
        // this function starts executing. This raises the lower limit by a bit
        // because by the time that this function is executing we've already
        // consumed at least a little bit of stack (we don't know the exact byte
        // address at which our stack started).
        let main = proc() {
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

            // There are two primary reasons that general try/catch is
            // unsafe. The first is that we do not support nested try/catch. The
            // fact that this is happening in a newly-spawned thread
            // suffices. The second is that unwinding while unwinding is not
            // defined.  We take care of that by having an 'unwinding' flag in
            // the thread itself. For these reasons, this unsafety should be ok.
            unsafe {
                let mut output = None;
                let f = if stdout.is_some() || stderr.is_some() {
                    proc() {
                        let _ = stdout.map(stdio::set_stdout);
                        let _ = stderr.map(stdio::set_stderr);
                        f()
                    }
                } else {
                    f
                };

                let try_result = {
                    let ptr = &mut output;
                    unwind::try(move || *ptr = Some(f()))
                };
                match (output, try_result) {
                    (Some(data), Ok(_)) => after(Ok(data)),
                    (None, Err(cause)) => after(Err(cause)),
                    _ => unreachable!()
                }
            }
        };
        (unsafe { imp::create(stack_size, box main) }, my_thread)
    }

    /// Spawn a detached thread, and return a handle to it.
    ///
    /// The new child thread may outlive its parent.
    pub fn spawn(self, f: proc():Send) -> Thread {
        let (native, thread) = self.core_spawn(f, proc(_) {});
        unsafe { imp::detach(native) };
        thread
    }

    /// Spawn a joinable thread, and return an RAII guard for it.
    pub fn with_join<T: Send>(self, f: proc():Send -> T) -> JoinGuard<T> {
        // We need the address of the packet to fill in to be stable so when
        // `main` fills it in it's still valid, so allocate an extra box to do
        // so.
        let any: Box<Any+Send> = box 0u8; // sentinel value
        let my_packet = box Err(any);
        let their_packet: *mut Result<T> = unsafe {
            *mem::transmute::<&Box<Result<T>>, *const *mut Result<T>>(&my_packet)
        };

        let (native, thread) = self.core_spawn(f, proc(result) {
            unsafe { *their_packet = result; }
        });

        JoinGuard {
            native: native,
            joined: false,
            packet: Some(my_packet),
            thread: thread,
        }
    }
}

/// A convenience function for creating configurations.
pub fn cfg() -> Cfg { Cfg::new() }

struct Inner {
    name: Option<String>,
    lock: Mutex<bool>,          // true when there is a buffered unpark
    cvar: Condvar,
}

#[deriving(Clone)]
/// A handle to a thread.
pub struct Thread {
    inner: Arc<Inner>,
}

impl Thread {
    fn new(name: Option<String>) -> Thread {
        Thread {
            inner: Arc::new(Inner {
                name: name,
                lock: Mutex::new(false),
                cvar: Condvar::new(),
            })
        }
    }

    /// Spawn a detached thread, and return a handle to it.
    ///
    /// The new child thread may outlive its parent.
    pub fn spawn(f: proc():Send) -> Thread {
        Cfg::new().spawn(f)
    }

    /// Spawn a joinable thread, and return an RAII guard for it.
    pub fn with_join<T: Send>(f: proc():Send -> T) -> JoinGuard<T> {
        Cfg::new().with_join(f)
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
        thread_info::panicking()
    }

    // http://cr.openjdk.java.net/~stefank/6989984.1/raw_files/new/src/os/linux/vm/os_linux.cpp
    /// Block unless or until the current thread's token is made available (may wake spuriously).
    ///
    /// See the module doc for more detail.
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
/// An RAII guard that will block until thread termination when dropped.
pub struct JoinGuard<T> {
    native: imp::rust_thread,
    thread: Thread,
    joined: bool,
    packet: Option<Box<Result<T>>>,
}

impl<T: Send> JoinGuard<T> {
    /// Extract a handle to the thread this guard will join on.
    pub fn thread(&self) -> Thread {
        self.thread.clone()
    }

    /// Wait for the associated thread to finish, returning the result of the thread's
    /// calculation.
    pub fn join(mut self) -> Result<T> {
        assert!(!self.joined);
        unsafe { imp::join(self.native) };
        self.joined = true;
        let box res =  self.packet.take().unwrap();
        res
    }
}

#[unsafe_destructor]
impl<T: Send> Drop for JoinGuard<T> {
    fn drop(&mut self) {
        // This is required for correctness. If this is not done then the thread
        // would fill in a return box which no longer exists.
        if !self.joined {
            unsafe { imp::join(self.native) };
        }
    }
}

// TODO: fix tests
#[cfg(test)]
mod test {
    use any::{Any, AnyRefExt};
    use boxed::BoxAny;
    use prelude::*;
    use result::Result::{Ok, Err};
    use result;
    use std::io::{ChanReader, ChanWriter};
    use string::String;
    use super::{Thread, cfg};

    // !!! These tests are dangerous. If something is buggy, they will hang, !!!
    // !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

    #[test]
    fn test_unnamed_thread() {
        Thread::with_join(proc() {
            assert!(Thread::current().name().is_none());
        }).join().map_err(|_| ()).unwrap();
    }

    #[test]
    fn test_named_thread() {
        cfg().name("ada lovelace".to_string()).with_join(proc() {
            assert!(Thread::current().name().unwrap() == "ada lovelace".to_string());
        }).join().map_err(|_| ()).unwrap();
    }

    #[test]
    fn test_run_basic() {
        let (tx, rx) = channel();
        Thread::spawn(proc() {
            tx.send(());
        });
        rx.recv();
    }

    #[test]
    fn test_join_success() {
        match Thread::with_join::<String>(proc() {
            "Success!".to_string()
        }).join().as_ref().map(|s| s.as_slice()) {
            result::Result::Ok("Success!") => (),
            _ => panic!()
        }
    }

    #[test]
    fn test_join_panic() {
        match Thread::with_join(proc() {
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
            Thread::spawn(proc() {
                if i == 0 {
                    tx.send(());
                } else {
                    f(i - 1, tx);
                }
            });

        }
        f(10, tx);
        rx.recv();
    }

    #[test]
    fn test_spawn_sched_childs_on_default_sched() {
        let (tx, rx) = channel();

        Thread::spawn(proc() {
            Thread::spawn(proc() {
                tx.send(());
            });
        });

        rx.recv();
    }

    fn avoid_copying_the_body(spawnfn: |v: proc():Send|) {
        let (tx, rx) = channel::<uint>();

        let x = box 1;
        let x_in_parent = (&*x) as *const int as uint;

        spawnfn(proc() {
            let x_in_child = (&*x) as *const int as uint;
            tx.send(x_in_child);
        });

        let x_in_child = rx.recv();
        assert_eq!(x_in_parent, x_in_child);
    }

    #[test]
    fn test_avoid_copying_the_body_spawn() {
        avoid_copying_the_body(|v| { Thread::spawn(v); });
    }

    #[test]
    fn test_avoid_copying_the_body_thread_spawn() {
        avoid_copying_the_body(|f| {
            let builder = cfg();
            builder.spawn(proc() {
                f();
            });
        })
    }

    #[test]
    fn test_avoid_copying_the_body_join() {
        avoid_copying_the_body(|f| {
            let _ = Thread::with_join(proc() {
                f()
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
        fn child_no(x: uint) -> proc(): Send {
            return proc() {
                if x < GENERATIONS {
                    Thread::spawn(child_no(x+1));
                }
            }
        }
        Thread::spawn(child_no(0));
    }

    #[test]
    fn test_simple_newsched_spawn() {
        Thread::spawn(proc()());
    }

    #[test]
    fn test_try_panic_message_static_str() {
        match Thread::with_join(proc() {
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
        match Thread::with_join(proc() {
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
        match Thread::with_join(proc() {
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

        match Thread::with_join(proc() {
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

        let r = cfg().stdout(box stdout as Box<Writer + Send>).with_join(proc() {
                print!("Hello, world!");
            }).join();
        assert!(r.is_ok());

        let output = reader.read_to_string().unwrap();
        assert_eq!(output, "Hello, world!".to_string());
    }

    // NOTE: the corresponding test for stderr is in run-pass/task-stderr, due
    // to the test harness apparently interfering with stderr configuration.
}
