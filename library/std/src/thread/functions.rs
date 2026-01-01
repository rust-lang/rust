//! Free functions.

use super::builder::Builder;
use super::current::current;
use super::join_handle::JoinHandle;
use crate::mem::forget;
use crate::num::NonZero;
use crate::sys::thread as imp;
use crate::time::{Duration, Instant};
use crate::{io, panicking};

/// Spawns a new thread, returning a [`JoinHandle`] for it.
///
/// The join handle provides a [`join`] method that can be used to join the spawned
/// thread. If the spawned thread panics, [`join`] will return an [`Err`] containing
/// the argument given to [`panic!`].
///
/// If the join handle is dropped, the spawned thread will implicitly be *detached*.
/// In this case, the spawned thread may no longer be joined.
/// (It is the responsibility of the program to either eventually join threads it
/// creates or detach them; otherwise, a resource leak will result.)
///
/// This function creates a thread with the default parameters of [`Builder`].
/// To specify the new thread's stack size or the name, use [`Builder::spawn`].
///
/// As you can see in the signature of `spawn` there are two constraints on
/// both the closure given to `spawn` and its return value, let's explain them:
///
/// - The `'static` constraint means that the closure and its return value
///   must have a lifetime of the whole program execution. The reason for this
///   is that threads can outlive the lifetime they have been created in.
///
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
///     println!("{value}");
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
/// println!("{result}");
/// ```
///
/// # Notes
///
/// This function has the same minimal guarantee regarding "foreign" unwinding operations (e.g.
/// an exception thrown from C++ code, or a `panic!` in Rust code compiled or linked with a
/// different runtime) as [`catch_unwind`]; namely, if the thread created with `thread::spawn`
/// unwinds all the way to the root with such an exception, one of two behaviors are possible,
/// and it is unspecified which will occur:
///
/// * The process aborts.
/// * The process does not abort, and [`join`] will return a `Result::Err`
///   containing an opaque type.
///
/// [`catch_unwind`]: ../../std/panic/fn.catch_unwind.html
/// [`channels`]: crate::sync::mpsc
/// [`join`]: JoinHandle::join
/// [`Err`]: crate::result::Result::Err
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(miri, track_caller)] // even without panics, this helps for Miri backtraces
pub fn spawn<F, T>(f: F) -> JoinHandle<T>
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    Builder::new().spawn(f).expect("failed to spawn thread")
}

/// Cooperatively gives up a timeslice to the OS scheduler.
///
/// This calls the underlying OS scheduler's yield primitive, signaling
/// that the calling thread is willing to give up its remaining timeslice
/// so that the OS may schedule other threads on the CPU.
///
/// A drawback of yielding in a loop is that if the OS does not have any
/// other ready threads to run on the current CPU, the thread will effectively
/// busy-wait, which wastes CPU time and energy.
///
/// Therefore, when waiting for events of interest, a programmer's first
/// choice should be to use synchronization devices such as [`channel`]s,
/// [`Condvar`]s, [`Mutex`]es or [`join`] since these primitives are
/// implemented in a blocking manner, giving up the CPU until the event
/// of interest has occurred which avoids repeated yielding.
///
/// `yield_now` should thus be used only rarely, mostly in situations where
/// repeated polling is required because there is no other suitable way to
/// learn when an event of interest has occurred.
///
/// # Examples
///
/// ```
/// use std::thread;
///
/// thread::yield_now();
/// ```
///
/// [`channel`]: crate::sync::mpsc
/// [`join`]: JoinHandle::join
/// [`Condvar`]: crate::sync::Condvar
/// [`Mutex`]: crate::sync::Mutex
#[stable(feature = "rust1", since = "1.0.0")]
pub fn yield_now() {
    imp::yield_now()
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
/// [Mutex]: crate::sync::Mutex
#[inline]
#[must_use]
#[stable(feature = "rust1", since = "1.0.0")]
pub fn panicking() -> bool {
    panicking::panicking()
}

/// Uses [`sleep`].
///
/// Puts the current thread to sleep for at least the specified amount of time.
///
/// The thread may sleep longer than the duration specified due to scheduling
/// specifics or platform-dependent functionality. It will never sleep less.
///
/// This function is blocking, and should not be used in `async` functions.
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
#[deprecated(since = "1.6.0", note = "replaced by `std::thread::sleep`")]
pub fn sleep_ms(ms: u32) {
    sleep(Duration::from_millis(ms as u64))
}

/// Puts the current thread to sleep for at least the specified amount of time.
///
/// The thread may sleep longer than the duration specified due to scheduling
/// specifics or platform-dependent functionality. It will never sleep less.
///
/// This function is blocking, and should not be used in `async` functions.
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
/// Currently, specifying a zero duration on Unix platforms returns immediately
/// without invoking the underlying [`nanosleep`] syscall, whereas on Windows
/// platforms the underlying [`Sleep`] syscall is always invoked.
/// If the intention is to yield the current time-slice you may want to use
/// [`yield_now`] instead.
///
/// [`nanosleep`]: https://linux.die.net/man/2/nanosleep
/// [`Sleep`]: https://docs.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-sleep
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
    imp::sleep(dur)
}

/// Puts the current thread to sleep until the specified deadline has passed.
///
/// The thread may still be asleep after the deadline specified due to
/// scheduling specifics or platform-dependent functionality. It will never
/// wake before.
///
/// This function is blocking, and should not be used in `async` functions.
///
/// # Platform-specific behavior
///
/// In most cases this function will call an OS specific function. Where that
/// is not supported [`sleep`] is used. Those platforms are referred to as other
/// in the table below.
///
/// # Underlying System calls
///
/// The following system calls are [currently] being used:
///
/// |  Platform |               System call                                            |
/// |-----------|----------------------------------------------------------------------|
/// | Linux     | [clock_nanosleep] (Monotonic clock)                                  |
/// | BSD except OpenBSD | [clock_nanosleep] (Monotonic Clock)]                        |
/// | Android   | [clock_nanosleep] (Monotonic Clock)]                                 |
/// | Solaris   | [clock_nanosleep] (Monotonic Clock)]                                 |
/// | Illumos   | [clock_nanosleep] (Monotonic Clock)]                                 |
/// | Dragonfly | [clock_nanosleep] (Monotonic Clock)]                                 |
/// | Hurd      | [clock_nanosleep] (Monotonic Clock)]                                 |
/// | Fuchsia   | [clock_nanosleep] (Monotonic Clock)]                                 |
/// | Vxworks   | [clock_nanosleep] (Monotonic Clock)]                                 |
/// | Other     | `sleep_until` uses [`sleep`] and does not issue a syscall itself     |
///
/// [currently]: crate::io#platform-specific-behavior
/// [clock_nanosleep]: https://linux.die.net/man/3/clock_nanosleep
///
/// **Disclaimer:** These system calls might change over time.
///
/// # Examples
///
/// A simple game loop that limits the game to 60 frames per second.
///
/// ```no_run
/// #![feature(thread_sleep_until)]
/// # use std::time::{Duration, Instant};
/// # use std::thread;
/// #
/// # fn update() {}
/// # fn render() {}
/// #
/// let max_fps = 60.0;
/// let frame_time = Duration::from_secs_f32(1.0/max_fps);
/// let mut next_frame = Instant::now();
/// loop {
///     thread::sleep_until(next_frame);
///     next_frame += frame_time;
///     update();
///     render();
/// }
/// ```
///
/// A slow API we must not call too fast and which takes a few
/// tries before succeeding. By using `sleep_until` the time the
/// API call takes does not influence when we retry or when we give up
///
/// ```no_run
/// #![feature(thread_sleep_until)]
/// # use std::time::{Duration, Instant};
/// # use std::thread;
/// #
/// # enum Status {
/// #     Ready(usize),
/// #     Waiting,
/// # }
/// # fn slow_web_api_call() -> Status { Status::Ready(42) }
/// #
/// # const MAX_DURATION: Duration = Duration::from_secs(10);
/// #
/// # fn try_api_call() -> Result<usize, ()> {
/// let deadline = Instant::now() + MAX_DURATION;
/// let delay = Duration::from_millis(250);
/// let mut next_attempt = Instant::now();
/// loop {
///     if Instant::now() > deadline {
///         break Err(());
///     }
///     if let Status::Ready(data) = slow_web_api_call() {
///         break Ok(data);
///     }
///
///     next_attempt = deadline.min(next_attempt + delay);
///     thread::sleep_until(next_attempt);
/// }
/// # }
/// # let _data = try_api_call();
/// ```
#[unstable(feature = "thread_sleep_until", issue = "113752")]
pub fn sleep_until(deadline: Instant) {
    imp::sleep_until(deadline)
}

/// Used to ensure that `park` and `park_timeout` do not unwind, as that can
/// cause undefined behavior if not handled correctly (see #102398 for context).
struct PanicGuard;

impl Drop for PanicGuard {
    fn drop(&mut self) {
        rtabort!("an irrecoverable error occurred while synchronizing threads")
    }
}

/// Blocks unless or until the current thread's token is made available.
///
/// A call to `park` does not guarantee that the thread will remain parked
/// forever, and callers should be prepared for this possibility. However,
/// it is guaranteed that this function will not panic (it may abort the
/// process if the implementation encounters some rare errors).
///
/// # `park` and `unpark`
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
///   if it wasn't already. Because the token can be held by a thread even if it is currently not
///   parked, [`unpark`] followed by [`park`] will result in the second call returning immediately.
///   However, note that to rely on this guarantee, you need to make sure that your `unpark` happens
///   after all `park` that may be done by other data structures!
///
/// The API is typically used by acquiring a handle to the current thread, placing that handle in a
/// shared data structure so that other threads can find it, and then `park`ing in a loop. When some
/// desired condition is met, another thread calls [`unpark`] on the handle. The last bullet point
/// above guarantees that even if the `unpark` occurs before the thread is finished `park`ing, it
/// will be woken up properly.
///
/// Note that the coordination via the shared data structure is crucial: If you `unpark` a thread
/// without first establishing that it is about to be `park`ing within your code, that `unpark` may
/// get consumed by a *different* `park` in the same thread, leading to a deadlock. This also means
/// you must not call unknown code between setting up for parking and calling `park`; for instance,
/// if you invoke `println!`, that may itself call `park` and thus consume your `unpark` and cause a
/// deadlock.
///
/// The motivation for this design is twofold:
///
/// * It avoids the need to allocate mutexes and condvars when building new
///   synchronization primitives; the threads already provide basic
///   blocking/signaling.
///
/// * It can be implemented very efficiently on many platforms.
///
/// # Memory Ordering
///
/// Calls to `unpark` _synchronize-with_ calls to `park`, meaning that memory
/// operations performed before a call to `unpark` are made visible to the thread that
/// consumes the token and returns from `park`. Note that all `park` and `unpark`
/// operations for a given thread form a total order and _all_ prior `unpark` operations
/// synchronize-with `park`.
///
/// In atomic ordering terms, `unpark` performs a `Release` operation and `park`
/// performs the corresponding `Acquire` operation. Calls to `unpark` for the same
/// thread form a [release sequence].
///
/// Note that being unblocked does not imply a call was made to `unpark`, because
/// wakeups can also be spurious. For example, a valid, but inefficient,
/// implementation could have `park` and `unpark` return immediately without doing anything,
/// making *all* wakeups spurious.
///
/// # Examples
///
/// ```
/// use std::thread;
/// use std::sync::atomic::{Ordering, AtomicBool};
/// use std::time::Duration;
///
/// static QUEUED: AtomicBool = AtomicBool::new(false);
/// static FLAG: AtomicBool = AtomicBool::new(false);
///
/// let parked_thread = thread::spawn(move || {
///     println!("Thread spawned");
///     // Signal that we are going to `park`. Between this store and our `park`, there may
///     // be no other `park`, or else that `park` could consume our `unpark` token!
///     QUEUED.store(true, Ordering::Release);
///     // We want to wait until the flag is set. We *could* just spin, but using
///     // park/unpark is more efficient.
///     while !FLAG.load(Ordering::Acquire) {
///         // We can *not* use `println!` here since that could use thread parking internally.
///         thread::park();
///         // We *could* get here spuriously, i.e., way before the 10ms below are over!
///         // But that is no problem, we are in a loop until the flag is set anyway.
///     }
///     println!("Flag received");
/// });
///
/// // Let some time pass for the thread to be spawned.
/// thread::sleep(Duration::from_millis(10));
///
/// // Ensure the thread is about to park.
/// // This is crucial! It guarantees that the `unpark` below is not consumed
/// // by some other code in the parked thread (e.g. inside `println!`).
/// while !QUEUED.load(Ordering::Acquire) {
///     // Spinning is of course inefficient; in practice, this would more likely be
///     // a dequeue where we have no work to do if there's nobody queued.
///     std::hint::spin_loop();
/// }
///
/// // Set the flag, and let the thread wake up.
/// // There is no race condition here: if `unpark`
/// // happens first, `park` will return immediately.
/// // There is also no other `park` that could consume this token,
/// // since we waited until the other thread got queued.
/// // Hence there is no risk of a deadlock.
/// FLAG.store(true, Ordering::Release);
/// println!("Unpark the thread");
/// parked_thread.thread().unpark();
///
/// parked_thread.join().unwrap();
/// ```
///
/// [`Thread`]: super::Thread
/// [`unpark`]: super::Thread::unpark
/// [`thread::park_timeout`]: park_timeout
/// [release sequence]: https://en.cppreference.com/w/cpp/atomic/memory_order#Release_sequence
#[stable(feature = "rust1", since = "1.0.0")]
pub fn park() {
    let guard = PanicGuard;
    // SAFETY: park_timeout is called on the parker owned by this thread.
    unsafe {
        current().park();
    }
    // No panic occurred, do not abort.
    forget(guard);
}

/// Uses [`park_timeout`].
///
/// Blocks unless or until the current thread's token is made available or
/// the specified duration has been reached (may wake spuriously).
///
/// The semantics of this function are equivalent to [`park`] except
/// that the thread will be blocked for roughly no longer than `dur`. This
/// method should not be used for precise timing due to anomalies such as
/// preemption or platform differences that might not cause the maximum
/// amount of time waited to be precisely `ms` long.
///
/// See the [park documentation][`park`] for more detail.
#[stable(feature = "rust1", since = "1.0.0")]
#[deprecated(since = "1.6.0", note = "replaced by `std::thread::park_timeout`")]
pub fn park_timeout_ms(ms: u32) {
    park_timeout(Duration::from_millis(ms as u64))
}

/// Blocks unless or until the current thread's token is made available or
/// the specified duration has been reached (may wake spuriously).
///
/// The semantics of this function are equivalent to [`park`][park] except
/// that the thread will be blocked for roughly no longer than `dur`. This
/// method should not be used for precise timing due to anomalies such as
/// preemption or platform differences that might not cause the maximum
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
///     println!("restarting park_timeout after {elapsed:?}");
///     timeout_remaining = timeout - elapsed;
/// }
/// ```
#[stable(feature = "park_timeout", since = "1.4.0")]
pub fn park_timeout(dur: Duration) {
    let guard = PanicGuard;
    // SAFETY: park_timeout is called on a handle owned by this thread.
    unsafe {
        current().park_timeout(dur);
    }
    // No panic occurred, do not abort.
    forget(guard);
}

/// Returns an estimate of the default amount of parallelism a program should use.
///
/// Parallelism is a resource. A given machine provides a certain capacity for
/// parallelism, i.e., a bound on the number of computations it can perform
/// simultaneously. This number often corresponds to the amount of CPUs a
/// computer has, but it may diverge in various cases.
///
/// Host environments such as VMs or container orchestrators may want to
/// restrict the amount of parallelism made available to programs in them. This
/// is often done to limit the potential impact of (unintentionally)
/// resource-intensive programs on other programs running on the same machine.
///
/// # Limitations
///
/// The purpose of this API is to provide an easy and portable way to query
/// the default amount of parallelism the program should use. Among other things it
/// does not expose information on NUMA regions, does not account for
/// differences in (co)processor capabilities or current system load,
/// and will not modify the program's global state in order to more accurately
/// query the amount of available parallelism.
///
/// Where both fixed steady-state and burst limits are available the steady-state
/// capacity will be used to ensure more predictable latencies.
///
/// Resource limits can be changed during the runtime of a program, therefore the value is
/// not cached and instead recomputed every time this function is called. It should not be
/// called from hot code.
///
/// The value returned by this function should be considered a simplified
/// approximation of the actual amount of parallelism available at any given
/// time. To get a more detailed or precise overview of the amount of
/// parallelism available to the program, you may wish to use
/// platform-specific APIs as well. The following platform limitations currently
/// apply to `available_parallelism`:
///
/// On Windows:
/// - It may undercount the amount of parallelism available on systems with more
///   than 64 logical CPUs. However, programs typically need specific support to
///   take advantage of more than 64 logical CPUs, and in the absence of such
///   support, the number returned by this function accurately reflects the
///   number of logical CPUs the program can use by default.
/// - It may overcount the amount of parallelism available on systems limited by
///   process-wide affinity masks, or job object limitations.
///
/// On Linux:
/// - It may overcount the amount of parallelism available when limited by a
///   process-wide affinity mask or cgroup quotas and `sched_getaffinity()` or cgroup fs can't be
///   queried, e.g. due to sandboxing.
/// - It may undercount the amount of parallelism if the current thread's affinity mask
///   does not reflect the process' cpuset, e.g. due to pinned threads.
/// - If the process is in a cgroup v1 cpu controller, this may need to
///   scan mountpoints to find the corresponding cgroup v1 controller,
///   which may take time on systems with large numbers of mountpoints.
///   (This does not apply to cgroup v2, or to processes not in a
///   cgroup.)
/// - It does not attempt to take `ulimit` into account. If there is a limit set on the number of
///   threads, `available_parallelism` cannot know how much of that limit a Rust program should
///   take, or know in a reliable and race-free way how much of that limit is already taken.
///
/// On all targets:
/// - It may overcount the amount of parallelism available when running in a VM
/// with CPU usage limits (e.g. an overcommitted host).
///
/// # Errors
///
/// This function will, but is not limited to, return errors in the following
/// cases:
///
/// - If the amount of parallelism is not known for the target platform.
/// - If the program lacks permission to query the amount of parallelism made
///   available to it.
///
/// # Examples
///
/// ```
/// # #![allow(dead_code)]
/// use std::{io, thread};
///
/// fn main() -> io::Result<()> {
///     let count = thread::available_parallelism()?.get();
///     assert!(count >= 1_usize);
///     Ok(())
/// }
/// ```
#[doc(alias = "available_concurrency")] // Alias for a previous name we gave this API on unstable.
#[doc(alias = "hardware_concurrency")] // Alias for C++ `std::thread::hardware_concurrency`.
#[doc(alias = "num_cpus")] // Alias for a popular ecosystem crate which provides similar functionality.
#[stable(feature = "available_parallelism", since = "1.59.0")]
pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    imp::available_parallelism()
}
