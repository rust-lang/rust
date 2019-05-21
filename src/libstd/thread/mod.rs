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
//! child thread (i.e., join).
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
//! In this example, the spawned thread is "detached" from the current
//! thread. This means that it can outlive its parent (the thread that spawned
//! it), unless this parent is the main thread.
//!
//! The parent thread can also wait on the completion of the child
//! thread; a call to [`spawn`] produces a [`JoinHandle`], which provides
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
//! The [`join`] method returns a [`thread::Result`] containing [`Ok`] of the final
//! value produced by the child thread, or [`Err`] of the value given to
//! a call to [`panic!`] if the child panicked.
//!
//! ## Configuring threads
//!
//! A new thread can be configured before it is spawned via the [`Builder`] type,
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
//! thread, use [`Thread::name`]. A couple examples of where the name of a thread gets used:
//!
//! * If a panic occurs in a named thread, the thread name will be printed in the panic message.
//! * The thread name is provided to the OS where applicable (e.g., `pthread_setname_np` in
//!   unix-like platforms).
//!
//! ## Stack size
//!
//! The default stack size for spawned threads is 2 MiB, though this particular stack size is
//! subject to change in the future. There are two ways to manually specify the stack size for
//! spawned threads:
//!
//! * Build the thread with [`Builder`] and pass the desired stack size to [`Builder::stack_size`].
//! * Set the `RUST_MIN_STACK` environment variable to an integer representing the desired stack
//!   size (in bytes). Note that setting [`Builder::stack_size`] will override this.
//!
//! Note that the stack size of the main thread is *not* determined by Rust.
//!
//! [channels]: ../../std/sync/mpsc/index.html
//! [`Arc`]: ../../std/sync/struct.Arc.html
//! [`spawn`]: ../../std/thread/fn.spawn.html
//! [`JoinHandle`]: ../../std/thread/struct.JoinHandle.html
//! [`JoinHandle::thread`]: ../../std/thread/struct.JoinHandle.html#method.thread
//! [`join`]: ../../std/thread/struct.JoinHandle.html#method.join
//! [`Result`]: ../../std/result/enum.Result.html
//! [`Ok`]: ../../std/result/enum.Result.html#variant.Ok
//! [`Err`]: ../../std/result/enum.Result.html#variant.Err
//! [`panic!`]: ../../std/macro.panic.html
//! [`Builder`]: ../../std/thread/struct.Builder.html
//! [`Builder::stack_size`]: ../../std/thread/struct.Builder.html#method.stack_size
//! [`Builder::name`]: ../../std/thread/struct.Builder.html#method.name
//! [`thread::current`]: ../../std/thread/fn.current.html
//! [`thread::Result`]: ../../std/thread/type.Result.html
//! [`Thread`]: ../../std/thread/struct.Thread.html
//! [`park`]: ../../std/thread/fn.park.html
//! [`unpark`]: ../../std/thread/struct.Thread.html#method.unpark
//! [`Thread::name`]: ../../std/thread/struct.Thread.html#method.name
//! [`thread::park_timeout`]: ../../std/thread/fn.park_timeout.html
//! [`Cell`]: ../cell/struct.Cell.html
//! [`RefCell`]: ../cell/struct.RefCell.html
//! [`thread_local!`]: ../macro.thread_local.html
//! [`with`]: struct.LocalKey.html#method.with

#![stable(feature = "rust1", since = "1.0.0")]

use crate::any::Any;
use crate::cell::UnsafeCell;
use crate::ffi::{CStr, CString};
use crate::fmt;
use crate::io;
use crate::mem;
use crate::num::NonZeroU64;
use crate::panic;
use crate::panicking;
use crate::str;
use crate::sync::{Mutex, Condvar, Arc};
use crate::sync::atomic::AtomicUsize;
use crate::sync::atomic::Ordering::SeqCst;
use crate::sys::thread as imp;
use crate::sys_common::mutex;
use crate::sys_common::thread_info;
use crate::sys_common::thread;
use crate::sys_common::{AsInner, IntoInner};
use crate::time::Duration;

////////////////////////////////////////////////////////////////////////////////
// Thread-local storage
////////////////////////////////////////////////////////////////////////////////

#[macro_use] mod local;

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::local::{LocalKey, AccessError};

// The types used by the thread_local! macro to access TLS keys. Note that there
// are two types, the "OS" type and the "fast" type. The OS thread local key
// type is accessed via platform-specific API calls and is slow, while the fast
// key type is accessed via code generated via LLVM, where TLS keys are set up
// by the elf linker. Note that the OS TLS type is always available: on macOS
// the standard library is compiled with support for older platform versions
// where fast TLS was not available; end-user code is compiled with fast TLS
// where available, but both are needed.

#[unstable(feature = "libstd_thread_internals", issue = "0")]
#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
#[doc(hidden)] pub use self::local::statik::Key as __StaticLocalKeyInner;
#[unstable(feature = "libstd_thread_internals", issue = "0")]
#[cfg(target_thread_local)]
#[doc(hidden)] pub use self::local::fast::Key as __FastLocalKeyInner;
#[unstable(feature = "libstd_thread_internals", issue = "0")]
#[doc(hidden)] pub use self::local::os::Key as __OsLocalKeyInner;

////////////////////////////////////////////////////////////////////////////////
// Builder
////////////////////////////////////////////////////////////////////////////////

/// Thread factory, which can be used in order to configure the properties of
/// a new thread.
///
/// Methods can be chained on it in order to configure it.
///
/// The two configurations available are:
///
/// - [`name`]: specifies an [associated name for the thread][naming-threads]
/// - [`stack_size`]: specifies the [desired stack size for the thread][stack-size]
///
/// The [`spawn`] method will take ownership of the builder and create an
/// [`io::Result`] to the thread handle with the given configuration.
///
/// The [`thread::spawn`] free function uses a `Builder` with default
/// configuration and [`unwrap`]s its return value.
///
/// You may want to use [`spawn`] instead of [`thread::spawn`], when you want
/// to recover from a failure to launch a thread, indeed the free function will
/// panic where the `Builder` method will return a [`io::Result`].
///
/// # Examples
///
/// ```
/// use std::thread;
///
/// let builder = thread::Builder::new();
///
/// let handler = builder.spawn(|| {
///     // thread code
/// }).unwrap();
///
/// handler.join().unwrap();
/// ```
///
/// [`thread::spawn`]: ../../std/thread/fn.spawn.html
/// [`stack_size`]: ../../std/thread/struct.Builder.html#method.stack_size
/// [`name`]: ../../std/thread/struct.Builder.html#method.name
/// [`spawn`]: ../../std/thread/struct.Builder.html#method.spawn
/// [`io::Result`]: ../../std/io/type.Result.html
/// [`unwrap`]: ../../std/result/enum.Result.html#method.unwrap
/// [naming-threads]: ./index.html#naming-threads
/// [stack-size]: ./index.html#stack-size
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Debug)]
pub struct Builder {
    // A name for the thread-to-be, for identification in panic messages
    name: Option<String>,
    // The size of the stack for the spawned thread in bytes
    stack_size: Option<usize>,
}

impl Builder {
    /// Generates the base configuration for spawning a thread, from which
    /// configuration methods can be chained.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new()
    ///                               .name("foo".into())
    ///                               .stack_size(32 * 1024);
    ///
    /// let handler = builder.spawn(|| {
    ///     // thread code
    /// }).unwrap();
    ///
    /// handler.join().unwrap();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn new() -> Builder {
        Builder {
            name: None,
            stack_size: None,
        }
    }

    /// Names the thread-to-be. Currently the name is used for identification
    /// only in panic messages.
    ///
    /// The name must not contain null bytes (`\0`).
    ///
    /// For more information about named threads, see
    /// [this module-level documentation][naming-threads].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new()
    ///     .name("foo".into());
    ///
    /// let handler = builder.spawn(|| {
    ///     assert_eq!(thread::current().name(), Some("foo"))
    /// }).unwrap();
    ///
    /// handler.join().unwrap();
    /// ```
    ///
    /// [naming-threads]: ./index.html#naming-threads
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn name(mut self, name: String) -> Builder {
        self.name = Some(name);
        self
    }

    /// Sets the size of the stack (in bytes) for the new thread.
    ///
    /// The actual stack size may be greater than this value if
    /// the platform specifies a minimal stack size.
    ///
    /// For more information about the stack size for threads, see
    /// [this module-level documentation][stack-size].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new().stack_size(32 * 1024);
    /// ```
    ///
    /// [stack-size]: ./index.html#stack-size
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn stack_size(mut self, size: usize) -> Builder {
        self.stack_size = Some(size);
        self
    }

    /// Spawns a new thread by taking ownership of the `Builder`, and returns an
    /// [`io::Result`] to its [`JoinHandle`].
    ///
    /// The spawned thread may outlive the caller (unless the caller thread
    /// is the main thread; the whole process is terminated when the main
    /// thread finishes). The join handle can be used to block on
    /// termination of the child thread, including recovering its panics.
    ///
    /// For a more complete documentation see [`thread::spawn`][`spawn`].
    ///
    /// # Errors
    ///
    /// Unlike the [`spawn`] free function, this method yields an
    /// [`io::Result`] to capture any failure to create the thread at
    /// the OS level.
    ///
    /// [`spawn`]: ../../std/thread/fn.spawn.html
    /// [`io::Result`]: ../../std/io/type.Result.html
    /// [`JoinHandle`]: ../../std/thread/struct.JoinHandle.html
    ///
    /// # Panics
    ///
    /// Panics if a thread name was set and it contained null bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new();
    ///
    /// let handler = builder.spawn(|| {
    ///     // thread code
    /// }).unwrap();
    ///
    /// handler.join().unwrap();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn spawn<F, T>(self, f: F) -> io::Result<JoinHandle<T>> where
        F: FnOnce() -> T, F: Send + 'static, T: Send + 'static
    {
        unsafe { self.spawn_unchecked(f) }
    }

    /// Spawns a new thread without any lifetime restrictions by taking ownership
    /// of the `Builder`, and returns an [`io::Result`] to its [`JoinHandle`].
    ///
    /// The spawned thread may outlive the caller (unless the caller thread
    /// is the main thread; the whole process is terminated when the main
    /// thread finishes). The join handle can be used to block on
    /// termination of the child thread, including recovering its panics.
    ///
    /// This method is identical to [`thread::Builder::spawn`][`Builder::spawn`],
    /// except for the relaxed lifetime bounds, which render it unsafe.
    /// For a more complete documentation see [`thread::spawn`][`spawn`].
    ///
    /// # Errors
    ///
    /// Unlike the [`spawn`] free function, this method yields an
    /// [`io::Result`] to capture any failure to create the thread at
    /// the OS level.
    ///
    /// # Panics
    ///
    /// Panics if a thread name was set and it contained null bytes.
    ///
    /// # Safety
    ///
    /// The caller has to ensure that no references in the supplied thread closure
    /// or its return type can outlive the spawned thread's lifetime. This can be
    /// guaranteed in two ways:
    ///
    /// - ensure that [`join`][`JoinHandle::join`] is called before any referenced
    /// data is dropped
    /// - use only types with `'static` lifetime bounds, i.e., those with no or only
    /// `'static` references (both [`thread::Builder::spawn`][`Builder::spawn`]
    /// and [`thread::spawn`][`spawn`] enforce this property statically)
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(thread_spawn_unchecked)]
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new();
    ///
    /// let x = 1;
    /// let thread_x = &x;
    ///
    /// let handler = unsafe {
    ///     builder.spawn_unchecked(move || {
    ///         println!("x = {}", *thread_x);
    ///     }).unwrap()
    /// };
    ///
    /// // caller has to ensure `join()` is called, otherwise
    /// // it is possible to access freed memory if `x` gets
    /// // dropped before the thread closure is executed!
    /// handler.join().unwrap();
    /// ```
    ///
    /// [`spawn`]: ../../std/thread/fn.spawn.html
    /// [`Builder::spawn`]: ../../std/thread/struct.Builder.html#method.spawn
    /// [`io::Result`]: ../../std/io/type.Result.html
    /// [`JoinHandle`]: ../../std/thread/struct.JoinHandle.html
    /// [`JoinHandle::join`]: ../../std/thread/struct.JoinHandle.html#method.join
    #[unstable(feature = "thread_spawn_unchecked", issue = "55132")]
    pub unsafe fn spawn_unchecked<'a, F, T>(self, f: F) -> io::Result<JoinHandle<T>> where
        F: FnOnce() -> T, F: Send + 'a, T: Send + 'a
    {
        let Builder { name, stack_size } = self;

        let stack_size = stack_size.unwrap_or_else(thread::min_stack);

        let my_thread = Thread::new(name);
        let their_thread = my_thread.clone();

        let my_packet : Arc<UnsafeCell<Option<Result<T>>>>
            = Arc::new(UnsafeCell::new(None));
        let their_packet = my_packet.clone();

        let main = move || {
            if let Some(name) = their_thread.cname() {
                imp::Thread::set_name(name);
            }

            thread_info::set(imp::guard::current(), their_thread);
            #[cfg(feature = "backtrace")]
            let try_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                crate::sys_common::backtrace::__rust_begin_short_backtrace(f)
            }));
            #[cfg(not(feature = "backtrace"))]
            let try_result = panic::catch_unwind(panic::AssertUnwindSafe(f));
            *their_packet.get() = Some(try_result);
        };

        Ok(JoinHandle(JoinInner {
            // `imp::Thread::new` takes a closure with a `'static` lifetime, since it's passed
            // through FFI or otherwise used with low-level threading primitives that have no
            // notion of or way to enforce lifetimes.
            //
            // As mentioned in the `Safety` section of this function's documentation, the caller of
            // this function needs to guarantee that the passed-in lifetime is sufficiently long
            // for the lifetime of the thread.
            //
            // Similarly, the `sys` implementation must guarantee that no references to the closure
            // exist after the thread has terminated, which is signaled by `Thread::join`
            // returning.
            native: Some(imp::Thread::new(
                stack_size,
                mem::transmute::<Box<dyn FnOnce() + 'a>, Box<dyn FnOnce() + 'static>>(Box::new(
                    main,
                )),
            )?),
            thread: my_thread,
            packet: Packet(my_packet),
        }))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Free functions
////////////////////////////////////////////////////////////////////////////////

/// Spawns a new thread, returning a [`JoinHandle`] for it.
///
/// The join handle will implicitly *detach* the child thread upon being
/// dropped. In this case, the child thread may outlive the parent (unless
/// the parent thread is the main thread; the whole process is terminated when
/// the main thread finishes). Additionally, the join handle provides a [`join`]
/// method that can be used to join the child thread. If the child thread
/// panics, [`join`] will return an [`Err`] containing the argument given to
/// [`panic`].
///
/// This will create a thread using default parameters of [`Builder`], if you
/// want to specify the stack size or the name of the thread, use this API
/// instead.
///
/// As you can see in the signature of `spawn` there are two constraints on
/// both the closure given to `spawn` and its return value, let's explain them:
///
/// - The `'static` constraint means that the closure and its return value
///   must have a lifetime of the whole program execution. The reason for this
///   is that threads can `detach` and outlive the lifetime they have been
///   created in.
///   Indeed if the thread, and by extension its return value, can outlive their
///   caller, we need to make sure that they will be valid afterwards, and since
///   we *can't* know when it will return we need to have them valid as long as
///   possible, that is until the end of the program, hence the `'static`
///   lifetime.
/// - The [`Send`] constraint is because the closure will need to be passed
///   *by value* from the thread where it is spawned to the new thread. Its
///   return value will need to be passed from the new thread to the thread
///   where it is `join`ed.
///   As a reminder, the [`Send`] marker trait expresses that it is safe to be
///   passed from thread to thread. [`Sync`] expresses that it is safe to have a
///   reference be passed from thread to thread.
///
/// # Panics
///
/// Panics if the OS fails to create a thread; use [`Builder::spawn`]
/// to recover from such errors.
///
/// # Examples
///
/// Creating a thread.
///
/// ```
/// use std::thread;
///
/// let handler = thread::spawn(|| {
///     // thread code
/// });
///
/// handler.join().unwrap();
/// ```
///
/// As mentioned in the module documentation, threads are usually made to
/// communicate using [`channels`], here is how it usually looks.
///
/// This example also shows how to use `move`, in order to give ownership
/// of values to a thread.
///
/// ```
/// use std::thread;
/// use std::sync::mpsc::channel;
///
/// let (tx, rx) = channel();
///
/// let sender = thread::spawn(move || {
///     tx.send("Hello, thread".to_owned())
///         .expect("Unable to send on channel");
/// });
///
/// let receiver = thread::spawn(move || {
///     let value = rx.recv().expect("Unable to receive from channel");
///     println!("{}", value);
/// });
///
/// sender.join().expect("The sender thread has panicked");
/// receiver.join().expect("The receiver thread has panicked");
/// ```
///
/// A thread can also return a value through its [`JoinHandle`], you can use
/// this to make asynchronous computations (futures might be more appropriate
/// though).
///
/// ```
/// use std::thread;
///
/// let computation = thread::spawn(|| {
///     // Some expensive computation.
///     42
/// });
///
/// let result = computation.join().unwrap();
/// println!("{}", result);
/// ```
///
/// [`channels`]: ../../std/sync/mpsc/index.html
/// [`JoinHandle`]: ../../std/thread/struct.JoinHandle.html
/// [`join`]: ../../std/thread/struct.JoinHandle.html#method.join
/// [`Err`]: ../../std/result/enum.Result.html#variant.Err
/// [`panic`]: ../../std/macro.panic.html
/// [`Builder::spawn`]: ../../std/thread/struct.Builder.html#method.spawn
/// [`Builder`]: ../../std/thread/struct.Builder.html
/// [`Send`]: ../../std/marker/trait.Send.html
/// [`Sync`]: ../../std/marker/trait.Sync.html
#[stable(feature = "rust1", since = "1.0.0")]
pub fn spawn<F, T>(f: F) -> JoinHandle<T> where
    F: FnOnce() -> T, F: Send + 'static, T: Send + 'static
{
    Builder::new().spawn(f).expect("failed to spawn thread")
}

/// Gets a handle to the thread that invokes it.
///
/// # Examples
///
/// Getting a handle to the current thread with `thread::current()`:
///
/// ```
/// use std::thread;
///
/// let handler = thread::Builder::new()
///     .name("named thread".into())
///     .spawn(|| {
///         let handle = thread::current();
///         assert_eq!(handle.name(), Some("named thread"));
///     })
///     .unwrap();
///
/// handler.join().unwrap();
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn current() -> Thread {
    thread_info::current_thread().expect("use of std::thread::current() is not \
                                          possible after the thread's local \
                                          data has been destroyed")
}

/// Cooperatively gives up a timeslice to the OS scheduler.
///
/// This is used when the programmer knows that the thread will have nothing
/// to do for some time, and thus avoid wasting computing time.
///
/// For example when polling on a resource, it is common to check that it is
/// available, and if not to yield in order to avoid busy waiting.
///
/// Thus the pattern of `yield`ing after a failed poll is rather common when
/// implementing low-level shared resources or synchronization primitives.
///
/// However programmers will usually prefer to use [`channel`]s, [`Condvar`]s,
/// [`Mutex`]es or [`join`] for their synchronization routines, as they avoid
/// thinking about thread scheduling.
///
/// Note that [`channel`]s for example are implemented using this primitive.
/// Indeed when you call `send` or `recv`, which are blocking, they will yield
/// if the channel is not available.
///
/// # Examples
///
/// ```
/// use std::thread;
///
/// thread::yield_now();
/// ```
///
/// [`channel`]: ../../std/sync/mpsc/index.html
/// [`spawn`]: ../../std/thread/fn.spawn.html
/// [`join`]: ../../std/thread/struct.JoinHandle.html#method.join
/// [`Mutex`]: ../../std/sync/struct.Mutex.html
/// [`Condvar`]: ../../std/sync/struct.Condvar.html
#[stable(feature = "rust1", since = "1.0.0")]
pub fn yield_now() {
    imp::Thread::yield_now()
}

/// Determines whether the current thread is unwinding because of panic.
///
/// A common use of this feature is to poison shared resources when writing
/// unsafe code, by checking `panicking` when the `drop` is called.
///
/// This is usually not needed when writing safe code, as [`Mutex`es][Mutex]
/// already poison themselves when a thread panics while holding the lock.
///
/// This can also be used in multithreaded applications, in order to send a
/// message to other threads warning that a thread has panicked (e.g., for
/// monitoring purposes).
///
/// # Examples
///
/// ```should_panic
/// use std::thread;
///
/// struct SomeStruct;
///
/// impl Drop for SomeStruct {
///     fn drop(&mut self) {
///         if thread::panicking() {
///             println!("dropped while unwinding");
///         } else {
///             println!("dropped while not unwinding");
///         }
///     }
/// }
///
/// {
///     print!("a: ");
///     let a = SomeStruct;
/// }
///
/// {
///     print!("b: ");
///     let b = SomeStruct;
///     panic!()
/// }
/// ```
///
/// [Mutex]: ../../std/sync/struct.Mutex.html
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn panicking() -> bool {
    panicking::panicking()
}

/// Puts the current thread to sleep for at least the specified amount of time.
///
/// The thread may sleep longer than the duration specified due to scheduling
/// specifics or platform-dependent functionality. It will never sleep less.
///
/// # Platform-specific behavior
///
/// On Unix platforms, the underlying syscall may be interrupted by a
/// spurious wakeup or signal handler. To ensure the sleep occurs for at least
/// the specified duration, this function may invoke that system call multiple
/// times.
///
/// # Examples
///
/// ```no_run
/// use std::thread;
///
/// // Let's sleep for 2 seconds:
/// thread::sleep_ms(2000);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(since = "1.6.0", reason = "replaced by `std::thread::sleep`")]
pub fn sleep_ms(ms: u32) {
    sleep(Duration::from_millis(ms as u64))
}

/// Puts the current thread to sleep for at least the specified amount of time.
///
/// The thread may sleep longer than the duration specified due to scheduling
/// specifics or platform-dependent functionality. It will never sleep less.
///
/// # Platform-specific behavior
///
/// On Unix platforms, the underlying syscall may be interrupted by a
/// spurious wakeup or signal handler. To ensure the sleep occurs for at least
/// the specified duration, this function may invoke that system call multiple
/// times.
/// Platforms which do not support nanosecond precision for sleeping will
/// have `dur` rounded up to the nearest granularity of time they can sleep for.
///
/// # Examples
///
/// ```no_run
/// use std::{thread, time};
///
/// let ten_millis = time::Duration::from_millis(10);
/// let now = time::Instant::now();
///
/// thread::sleep(ten_millis);
///
/// assert!(now.elapsed() >= ten_millis);
/// ```
#[stable(feature = "thread_sleep", since = "1.4.0")]
pub fn sleep(dur: Duration) {
    imp::Thread::sleep(dur)
}

// constants for park/unpark
const EMPTY: usize = 0;
const PARKED: usize = 1;
const NOTIFIED: usize = 2;

/// Blocks unless or until the current thread's token is made available.
///
/// A call to `park` does not guarantee that the thread will remain parked
/// forever, and callers should be prepared for this possibility.
///
/// # park and unpark
///
/// Every thread is equipped with some basic low-level blocking support, via the
/// [`thread::park`][`park`] function and [`thread::Thread::unpark`][`unpark`]
/// method. [`park`] blocks the current thread, which can then be resumed from
/// another thread by calling the [`unpark`] method on the blocked thread's
/// handle.
///
/// Conceptually, each [`Thread`] handle has an associated token, which is
/// initially not present:
///
/// * The [`thread::park`][`park`] function blocks the current thread unless or
///   until the token is available for its thread handle, at which point it
///   atomically consumes the token. It may also return *spuriously*, without
///   consuming the token. [`thread::park_timeout`] does the same, but allows
///   specifying a maximum time to block the thread for.
///
/// * The [`unpark`] method on a [`Thread`] atomically makes the token available
///   if it wasn't already. Because the token is initially absent, [`unpark`]
///   followed by [`park`] will result in the second call returning immediately.
///
/// In other words, each [`Thread`] acts a bit like a spinlock that can be
/// locked and unlocked using `park` and `unpark`.
///
/// Notice that being unblocked does not imply any synchronization with someone
/// that unparked this thread, it could also be spurious.
/// For example, it would be a valid, but inefficient, implementation to make both [`park`] and
/// [`unpark`] return immediately without doing anything.
///
/// The API is typically used by acquiring a handle to the current thread,
/// placing that handle in a shared data structure so that other threads can
/// find it, and then `park`ing in a loop. When some desired condition is met, another
/// thread calls [`unpark`] on the handle.
///
/// The motivation for this design is twofold:
///
/// * It avoids the need to allocate mutexes and condvars when building new
///   synchronization primitives; the threads already provide basic
///   blocking/signaling.
///
/// * It can be implemented very efficiently on many platforms.
///
/// # Examples
///
/// ```
/// use std::thread;
/// use std::sync::{Arc, atomic::{Ordering, AtomicBool}};
/// use std::time::Duration;
///
/// let flag = Arc::new(AtomicBool::new(false));
/// let flag2 = Arc::clone(&flag);
///
/// let parked_thread = thread::spawn(move || {
///     // We want to wait until the flag is set. We *could* just spin, but using
///     // park/unpark is more efficient.
///     while !flag2.load(Ordering::Acquire) {
///         println!("Parking thread");
///         thread::park();
///         // We *could* get here spuriously, i.e., way before the 10ms below are over!
///         // But that is no problem, we are in a loop until the flag is set anyway.
///         println!("Thread unparked");
///     }
///     println!("Flag received");
/// });
///
/// // Let some time pass for the thread to be spawned.
/// thread::sleep(Duration::from_millis(10));
///
/// // Set the flag, and let the thread wake up.
/// // There is no race condition here, if `unpark`
/// // happens first, `park` will return immediately.
/// // Hence there is no risk of a deadlock.
/// flag.store(true, Ordering::Release);
/// println!("Unpark the thread");
/// parked_thread.thread().unpark();
///
/// parked_thread.join().unwrap();
/// ```
///
/// [`Thread`]: ../../std/thread/struct.Thread.html
/// [`park`]: ../../std/thread/fn.park.html
/// [`unpark`]: ../../std/thread/struct.Thread.html#method.unpark
/// [`thread::park_timeout`]: ../../std/thread/fn.park_timeout.html
//
// The implementation currently uses the trivial strategy of a Mutex+Condvar
// with wakeup flag, which does not actually allow spurious wakeups. In the
// future, this will be implemented in a more efficient way, perhaps along the lines of
//   http://cr.openjdk.java.net/~stefank/6989984.1/raw_files/new/src/os/linux/vm/os_linux.cpp
// or futuxes, and in either case may allow spurious wakeups.
#[stable(feature = "rust1", since = "1.0.0")]
pub fn park() {
    let thread = current();

    // If we were previously notified then we consume this notification and
    // return quickly.
    if thread.inner.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst).is_ok() {
        return
    }

    // Otherwise we need to coordinate going to sleep
    let mut m = thread.inner.lock.lock().unwrap();
    match thread.inner.state.compare_exchange(EMPTY, PARKED, SeqCst, SeqCst) {
        Ok(_) => {}
        Err(NOTIFIED) => {
            // We must read here, even though we know it will be `NOTIFIED`.
            // This is because `unpark` may have been called again since we read
            // `NOTIFIED` in the `compare_exchange` above. We must perform an
            // acquire operation that synchronizes with that `unpark` to observe
            // any writes it made before the call to unpark. To do that we must
            // read from the write it made to `state`.
            let old = thread.inner.state.swap(EMPTY, SeqCst);
            assert_eq!(old, NOTIFIED, "park state changed unexpectedly");
            return;
        } // should consume this notification, so prohibit spurious wakeups in next park.
        Err(_) => panic!("inconsistent park state"),
    }
    loop {
        m = thread.inner.cvar.wait(m).unwrap();
        match thread.inner.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst) {
            Ok(_) => return, // got a notification
            Err(_) => {} // spurious wakeup, go back to sleep
        }
    }
}

/// Use [`park_timeout`].
///
/// Blocks unless or until the current thread's token is made available or
/// the specified duration has been reached (may wake spuriously).
///
/// The semantics of this function are equivalent to [`park`] except
/// that the thread will be blocked for roughly no longer than `dur`. This
/// method should not be used for precise timing due to anomalies such as
/// preemption or platform differences that may not cause the maximum
/// amount of time waited to be precisely `ms` long.
///
/// See the [park documentation][`park`] for more detail.
///
/// [`park_timeout`]: fn.park_timeout.html
/// [`park`]: ../../std/thread/fn.park.html
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(since = "1.6.0", reason = "replaced by `std::thread::park_timeout`")]
pub fn park_timeout_ms(ms: u32) {
    park_timeout(Duration::from_millis(ms as u64))
}

/// Blocks unless or until the current thread's token is made available or
/// the specified duration has been reached (may wake spuriously).
///
/// The semantics of this function are equivalent to [`park`][park] except
/// that the thread will be blocked for roughly no longer than `dur`. This
/// method should not be used for precise timing due to anomalies such as
/// preemption or platform differences that may not cause the maximum
/// amount of time waited to be precisely `dur` long.
///
/// See the [park documentation][park] for more details.
///
/// # Platform-specific behavior
///
/// Platforms which do not support nanosecond precision for sleeping will have
/// `dur` rounded up to the nearest granularity of time they can sleep for.
///
/// # Examples
///
/// Waiting for the complete expiration of the timeout:
///
/// ```rust,no_run
/// use std::thread::park_timeout;
/// use std::time::{Instant, Duration};
///
/// let timeout = Duration::from_secs(2);
/// let beginning_park = Instant::now();
///
/// let mut timeout_remaining = timeout;
/// loop {
///     park_timeout(timeout_remaining);
///     let elapsed = beginning_park.elapsed();
///     if elapsed >= timeout {
///         break;
///     }
///     println!("restarting park_timeout after {:?}", elapsed);
///     timeout_remaining = timeout - elapsed;
/// }
/// ```
///
/// [park]: fn.park.html
#[stable(feature = "park_timeout", since = "1.4.0")]
pub fn park_timeout(dur: Duration) {
    let thread = current();

    // Like `park` above we have a fast path for an already-notified thread, and
    // afterwards we start coordinating for a sleep.
    // return quickly.
    if thread.inner.state.compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst).is_ok() {
        return
    }
    let m = thread.inner.lock.lock().unwrap();
    match thread.inner.state.compare_exchange(EMPTY, PARKED, SeqCst, SeqCst) {
        Ok(_) => {}
        Err(NOTIFIED) => {
            // We must read again here, see `park`.
            let old = thread.inner.state.swap(EMPTY, SeqCst);
            assert_eq!(old, NOTIFIED, "park state changed unexpectedly");
            return;
        } // should consume this notification, so prohibit spurious wakeups in next park.
        Err(_) => panic!("inconsistent park_timeout state"),
    }

    // Wait with a timeout, and if we spuriously wake up or otherwise wake up
    // from a notification we just want to unconditionally set the state back to
    // empty, either consuming a notification or un-flagging ourselves as
    // parked.
    let (_m, _result) = thread.inner.cvar.wait_timeout(m, dur).unwrap();
    match thread.inner.state.swap(EMPTY, SeqCst) {
        NOTIFIED => {} // got a notification, hurray!
        PARKED => {} // no notification, alas
        n => panic!("inconsistent park_timeout state: {}", n),
    }
}

////////////////////////////////////////////////////////////////////////////////
// ThreadId
////////////////////////////////////////////////////////////////////////////////

/// A unique identifier for a running thread.
///
/// A `ThreadId` is an opaque object that has a unique value for each thread
/// that creates one. `ThreadId`s are not guaranteed to correspond to a thread's
/// system-designated identifier. A `ThreadId` can be retrieved from the [`id`]
/// method on a [`Thread`].
///
/// # Examples
///
/// ```
/// use std::thread;
///
/// let other_thread = thread::spawn(|| {
///     thread::current().id()
/// });
///
/// let other_thread_id = other_thread.join().unwrap();
/// assert!(thread::current().id() != other_thread_id);
/// ```
///
/// [`id`]: ../../std/thread/struct.Thread.html#method.id
/// [`Thread`]: ../../std/thread/struct.Thread.html
#[stable(feature = "thread_id", since = "1.19.0")]
#[derive(Eq, PartialEq, Clone, Copy, Hash, Debug)]
pub struct ThreadId(NonZeroU64);

impl ThreadId {
    // Generate a new unique thread ID.
    fn new() -> ThreadId {
        // We never call `GUARD.init()`, so it is UB to attempt to
        // acquire this mutex reentrantly!
        static GUARD: mutex::Mutex = mutex::Mutex::new();
        static mut COUNTER: u64 = 1;

        unsafe {
            let _guard = GUARD.lock();

            // If we somehow use up all our bits, panic so that we're not
            // covering up subtle bugs of IDs being reused.
            if COUNTER == crate::u64::MAX {
                panic!("failed to generate unique thread ID: bitspace exhausted");
            }

            let id = COUNTER;
            COUNTER += 1;

            ThreadId(NonZeroU64::new(id).unwrap())
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Thread
////////////////////////////////////////////////////////////////////////////////

/// The internal representation of a `Thread` handle
struct Inner {
    name: Option<CString>,      // Guaranteed to be UTF-8
    id: ThreadId,

    // state for thread park/unpark
    state: AtomicUsize,
    lock: Mutex<()>,
    cvar: Condvar,
}

#[derive(Clone)]
#[stable(feature = "rust1", since = "1.0.0")]
/// A handle to a thread.
///
/// Threads are represented via the `Thread` type, which you can get in one of
/// two ways:
///
/// * By spawning a new thread, e.g., using the [`thread::spawn`][`spawn`]
///   function, and calling [`thread`][`JoinHandle::thread`] on the
///   [`JoinHandle`].
/// * By requesting the current thread, using the [`thread::current`] function.
///
/// The [`thread::current`] function is available even for threads not spawned
/// by the APIs of this module.
///
/// There is usually no need to create a `Thread` struct yourself, one
/// should instead use a function like `spawn` to create new threads, see the
/// docs of [`Builder`] and [`spawn`] for more details.
///
/// [`Builder`]: ../../std/thread/struct.Builder.html
/// [`JoinHandle::thread`]: ../../std/thread/struct.JoinHandle.html#method.thread
/// [`JoinHandle`]: ../../std/thread/struct.JoinHandle.html
/// [`thread::current`]: ../../std/thread/fn.current.html
/// [`spawn`]: ../../std/thread/fn.spawn.html

pub struct Thread {
    inner: Arc<Inner>,
}

impl Thread {
    // Used only internally to construct a thread object without spawning
    // Panics if the name contains nuls.
    pub(crate) fn new(name: Option<String>) -> Thread {
        let cname = name.map(|n| {
            CString::new(n).expect("thread name may not contain interior null bytes")
        });
        Thread {
            inner: Arc::new(Inner {
                name: cname,
                id: ThreadId::new(),
                state: AtomicUsize::new(EMPTY),
                lock: Mutex::new(()),
                cvar: Condvar::new(),
            })
        }
    }

    /// Atomically makes the handle's token available if it is not already.
    ///
    /// Every thread is equipped with some basic low-level blocking support, via
    /// the [`park`][park] function and the `unpark()` method. These can be
    /// used as a more CPU-efficient implementation of a spinlock.
    ///
    /// See the [park documentation][park] for more details.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// let parked_thread = thread::Builder::new()
    ///     .spawn(|| {
    ///         println!("Parking thread");
    ///         thread::park();
    ///         println!("Thread unparked");
    ///     })
    ///     .unwrap();
    ///
    /// // Let some time pass for the thread to be spawned.
    /// thread::sleep(Duration::from_millis(10));
    ///
    /// println!("Unpark the thread");
    /// parked_thread.thread().unpark();
    ///
    /// parked_thread.join().unwrap();
    /// ```
    ///
    /// [park]: fn.park.html
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn unpark(&self) {
        // To ensure the unparked thread will observe any writes we made
        // before this call, we must perform a release operation that `park`
        // can synchronize with. To do that we must write `NOTIFIED` even if
        // `state` is already `NOTIFIED`. That is why this must be a swap
        // rather than a compare-and-swap that returns if it reads `NOTIFIED`
        // on failure.
        match self.inner.state.swap(NOTIFIED, SeqCst) {
            EMPTY => return, // no one was waiting
            NOTIFIED => return, // already unparked
            PARKED => {} // gotta go wake someone up
            _ => panic!("inconsistent state in unpark"),
        }

        // There is a period between when the parked thread sets `state` to
        // `PARKED` (or last checked `state` in the case of a spurious wake
        // up) and when it actually waits on `cvar`. If we were to notify
        // during this period it would be ignored and then when the parked
        // thread went to sleep it would never wake up. Fortunately, it has
        // `lock` locked at this stage so we can acquire `lock` to wait until
        // it is ready to receive the notification.
        //
        // Releasing `lock` before the call to `notify_one` means that when the
        // parked thread wakes it doesn't get woken only to have to wait for us
        // to release `lock`.
        drop(self.inner.lock.lock().unwrap());
        self.inner.cvar.notify_one()
    }

    /// Gets the thread's unique identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// let other_thread = thread::spawn(|| {
    ///     thread::current().id()
    /// });
    ///
    /// let other_thread_id = other_thread.join().unwrap();
    /// assert!(thread::current().id() != other_thread_id);
    /// ```
    #[stable(feature = "thread_id", since = "1.19.0")]
    pub fn id(&self) -> ThreadId {
        self.inner.id
    }

    /// Gets the thread's name.
    ///
    /// For more information about named threads, see
    /// [this module-level documentation][naming-threads].
    ///
    /// # Examples
    ///
    /// Threads by default have no name specified:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new();
    ///
    /// let handler = builder.spawn(|| {
    ///     assert!(thread::current().name().is_none());
    /// }).unwrap();
    ///
    /// handler.join().unwrap();
    /// ```
    ///
    /// Thread with a specified name:
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new()
    ///     .name("foo".into());
    ///
    /// let handler = builder.spawn(|| {
    ///     assert_eq!(thread::current().name(), Some("foo"))
    /// }).unwrap();
    ///
    /// handler.join().unwrap();
    /// ```
    ///
    /// [naming-threads]: ./index.html#naming-threads
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn name(&self) -> Option<&str> {
        self.cname().map(|s| unsafe { str::from_utf8_unchecked(s.to_bytes()) } )
    }

    fn cname(&self) -> Option<&CStr> {
        self.inner.name.as_ref().map(|s| &**s)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Debug for Thread {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Thread")
            .field("id", &self.id())
            .field("name", &self.name())
            .finish()
    }
}

////////////////////////////////////////////////////////////////////////////////
// JoinHandle
////////////////////////////////////////////////////////////////////////////////

/// A specialized [`Result`] type for threads.
///
/// Indicates the manner in which a thread exited.
///
/// A thread that completes without panicking is considered to exit successfully.
///
/// # Examples
///
/// ```no_run
/// use std::thread;
/// use std::fs;
///
/// fn copy_in_thread() -> thread::Result<()> {
///     thread::spawn(move || { fs::copy("foo.txt", "bar.txt").unwrap(); }).join()
/// }
///
/// fn main() {
///     match copy_in_thread() {
///         Ok(_) => println!("this is fine"),
///         Err(_) => println!("thread panicked"),
///     }
/// }
/// ```
///
/// [`Result`]: ../../std/result/enum.Result.html
#[stable(feature = "rust1", since = "1.0.0")]
pub type Result<T> = crate::result::Result<T, Box<dyn Any + Send + 'static>>;

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
/// A `JoinHandle` *detaches* the associated thread when it is dropped, which
/// means that there is no longer any handle to thread and no way to `join`
/// on it.
///
/// Due to platform restrictions, it is not possible to [`Clone`] this
/// handle: the ability to join a thread is a uniquely-owned permission.
///
/// This `struct` is created by the [`thread::spawn`] function and the
/// [`thread::Builder::spawn`] method.
///
/// # Examples
///
/// Creation from [`thread::spawn`]:
///
/// ```
/// use std::thread;
///
/// let join_handle: thread::JoinHandle<_> = thread::spawn(|| {
///     // some work here
/// });
/// ```
///
/// Creation from [`thread::Builder::spawn`]:
///
/// ```
/// use std::thread;
///
/// let builder = thread::Builder::new();
///
/// let join_handle: thread::JoinHandle<_> = builder.spawn(|| {
///     // some work here
/// }).unwrap();
/// ```
///
/// Child being detached and outliving its parent:
///
/// ```no_run
/// use std::thread;
/// use std::time::Duration;
///
/// let original_thread = thread::spawn(|| {
///     let _detached_thread = thread::spawn(|| {
///         // Here we sleep to make sure that the first thread returns before.
///         thread::sleep(Duration::from_millis(10));
///         // This will be called, even though the JoinHandle is dropped.
///         println!("♫ Still alive ♫");
///     });
/// });
///
/// original_thread.join().expect("The thread being joined has panicked");
/// println!("Original thread is joined.");
///
/// // We make sure that the new thread has time to run, before the main
/// // thread returns.
///
/// thread::sleep(Duration::from_millis(1000));
/// ```
///
/// [`Clone`]: ../../std/clone/trait.Clone.html
/// [`thread::spawn`]: fn.spawn.html
/// [`thread::Builder::spawn`]: struct.Builder.html#method.spawn
#[stable(feature = "rust1", since = "1.0.0")]
pub struct JoinHandle<T>(JoinInner<T>);

#[stable(feature = "joinhandle_impl_send_sync", since = "1.29.0")]
unsafe impl<T> Send for JoinHandle<T> {}
#[stable(feature = "joinhandle_impl_send_sync", since = "1.29.0")]
unsafe impl<T> Sync for JoinHandle<T> {}

impl<T> JoinHandle<T> {
    /// Extracts a handle to the underlying thread.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new();
    ///
    /// let join_handle: thread::JoinHandle<_> = builder.spawn(|| {
    ///     // some work here
    /// }).unwrap();
    ///
    /// let thread = join_handle.thread();
    /// println!("thread id: {:?}", thread.id());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn thread(&self) -> &Thread {
        &self.0.thread
    }

    /// Waits for the associated thread to finish.
    ///
    /// In terms of [atomic memory orderings],  the completion of the associated
    /// thread synchronizes with this function returning. In other words, all
    /// operations performed by that thread are ordered before all
    /// operations that happen after `join` returns.
    ///
    /// If the child thread panics, [`Err`] is returned with the parameter given
    /// to [`panic`].
    ///
    /// [`Err`]: ../../std/result/enum.Result.html#variant.Err
    /// [`panic`]: ../../std/macro.panic.html
    /// [atomic memory orderings]: ../../std/sync/atomic/index.html
    ///
    /// # Panics
    ///
    /// This function may panic on some platforms if a thread attempts to join
    /// itself or otherwise may create a deadlock with joining threads.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// let builder = thread::Builder::new();
    ///
    /// let join_handle: thread::JoinHandle<_> = builder.spawn(|| {
    ///     // some work here
    /// }).unwrap();
    /// join_handle.join().expect("Couldn't join on the associated thread");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn join(mut self) -> Result<T> {
        self.0.join()
    }
}

impl<T> AsInner<imp::Thread> for JoinHandle<T> {
    fn as_inner(&self) -> &imp::Thread { self.0.native.as_ref().unwrap() }
}

impl<T> IntoInner<imp::Thread> for JoinHandle<T> {
    fn into_inner(self) -> imp::Thread { self.0.native.unwrap() }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T> fmt::Debug for JoinHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad("JoinHandle { .. }")
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

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use super::Builder;
    use crate::any::Any;
    use crate::mem;
    use crate::sync::mpsc::{channel, Sender};
    use crate::result;
    use crate::thread::{self, ThreadId};
    use crate::time::Duration;
    use crate::u32;

    // !!! These tests are dangerous. If something is buggy, they will hang, !!!
    // !!! instead of exiting cleanly. This might wedge the buildbots.       !!!

    #[test]
    fn test_unnamed_thread() {
        thread::spawn(move|| {
            assert!(thread::current().name().is_none());
        }).join().ok().expect("thread panicked");
    }

    #[test]
    fn test_named_thread() {
        Builder::new().name("ada lovelace".to_string()).spawn(move|| {
            assert!(thread::current().name().unwrap() == "ada lovelace".to_string());
        }).unwrap().join().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_invalid_named_thread() {
        let _ = Builder::new().name("ada l\0velace".to_string()).spawn(|| {});
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

    fn avoid_copying_the_body<F>(spawnfn: F) where F: FnOnce(Box<dyn Fn() + Send>) {
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
        fn child_no(x: u32) -> Box<dyn Fn() + Send> {
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
            panic!(box 413u16 as Box<dyn Any + Send>);
        }).join() {
            Err(e) => {
                type T = Box<dyn Any + Send>;
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
            thread::park_timeout(Duration::from_millis(u32::MAX as u64));
        }
    }

    #[test]
    #[cfg_attr(target_env = "sgx", ignore)] // FIXME: https://github.com/fortanix/rust-sgx/issues/31
    fn test_park_timeout_unpark_not_called() {
        for _ in 0..10 {
            thread::park_timeout(Duration::from_millis(10));
        }
    }

    #[test]
    #[cfg_attr(target_env = "sgx", ignore)] // FIXME: https://github.com/fortanix/rust-sgx/issues/31
    fn test_park_timeout_unpark_called_other_thread() {
        for _ in 0..10 {
            let th = thread::current();

            let _guard = thread::spawn(move || {
                super::sleep(Duration::from_millis(50));
                th.unpark();
            });

            thread::park_timeout(Duration::from_millis(u32::MAX as u64));
        }
    }

    #[test]
    #[cfg_attr(target_env = "sgx", ignore)] // FIXME: https://github.com/fortanix/rust-sgx/issues/31
    fn sleep_ms_smoke() {
        thread::sleep(Duration::from_millis(2));
    }

    #[test]
    fn test_size_of_option_thread_id() {
        assert_eq!(mem::size_of::<Option<ThreadId>>(), mem::size_of::<ThreadId>());
    }

    #[test]
    fn test_thread_id_equal() {
        assert!(thread::current().id() == thread::current().id());
    }

    #[test]
    fn test_thread_id_not_equal() {
        let spawned_id = thread::spawn(|| thread::current().id()).join().unwrap();
        assert!(thread::current().id() != spawned_id);
    }

    // NOTE: the corresponding test for stderr is in run-pass/thread-stderr, due
    // to the test harness apparently interfering with stderr configuration.
}
