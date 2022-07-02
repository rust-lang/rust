use super::{current, park, Builder, JoinInner, Result, Thread};
use crate::cell::UnsafeCell;
use crate::fmt;
use crate::io;
use crate::marker::{PhantomData, PhantomPinned};
use crate::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use crate::pin::Pin;
use crate::ptr::NonNull;
use crate::sync::atomic::{fence, AtomicBool, AtomicUsize, Ordering};
use crate::sync::Arc;
use core::intrinsics::{atomic_store_rel, atomic_xsub_rel};

/// A scope to spawn scoped threads in.
///
/// See [`scope`] for details.
#[stable(feature = "scoped_threads", since = "1.63.0")]
pub struct Scope<'scope, 'env: 'scope> {
    data: Pin<&'scope ScopeData>,
    /// Invariance over 'scope, to make sure 'scope cannot shrink,
    /// which is necessary for soundness.
    ///
    /// Without invariance, this would compile fine but be unsound:
    ///
    /// ```compile_fail,E0373
    /// std::thread::scope(|s| {
    ///     s.spawn(|| {
    ///         let a = String::from("abcd");
    ///         s.spawn(|| println!("{a:?}")); // might run after `a` is dropped
    ///     });
    /// });
    /// ```
    scope: PhantomData<&'scope mut &'scope ()>,
    env: PhantomData<&'env mut &'env ()>,
}

/// An owned permission to join on a scoped thread (block on its termination).
///
/// See [`Scope::spawn`] for details.
#[stable(feature = "scoped_threads", since = "1.63.0")]
pub struct ScopedJoinHandle<'scope, T>(JoinInner<'scope, T>);

const WAITING_BIT: usize = 1;
const ONE_RUNNING: usize = 2;

/// Artificial limit on the maximum number of concurrently running threads in scope.
/// This is used to preemptively avoid hitting an overflow condition in the running thread count.
const MAX_RUNNING: usize = usize::MAX / 2;

#[derive(Default)]
pub(super) struct ScopeData {
    sync_state: AtomicUsize,
    thread_panicked: AtomicBool,
    scope_thread: UnsafeCell<Option<Thread>>,
    _pinned: PhantomPinned,
}

unsafe impl Send for ScopeData {} // SAFETY: ScopeData needs to be sent to the spawned threads in the scope.
unsafe impl Sync for ScopeData {} // SAFETY: ScopeData is shared between the spawned threads and the scope thread.

impl ScopeData {
    /// Issues an Acquire fence which synchronizes with the `sync_state` Release sequence.
    fn fence_acquire_sync_state(&self) {
        // ThreadSanitizier doesn't properly support fences
        // so use an atomic load instead to avoid false positive data-race reports.
        if cfg!(sanitize = "thread") {
            self.sync_state.load(Ordering::Acquire);
        } else {
            fence(Ordering::Acquire);
        }
    }

    pub(super) fn increment_num_running_threads(&self) {
        // No need for any memory barriers as this is just incrementing the running count
        // with the assumption that the ScopeData remains valid before and after this call.
        let state = self.sync_state.fetch_add(ONE_RUNNING, Ordering::Relaxed);

        // Make sure we're not spawning too many threads on the scope.
        // The `MAX_RUNNING` is intentionally lower than `usize::MAX` to detect overflow
        // conditions on the running count earlier, even in the presence of multiple threads.
        let running_threads = state / ONE_RUNNING;
        assert!(running_threads <= MAX_RUNNING, "too many running threads in thread scope");
    }

    /// Decrement the number of running threads with the assumption that one was running before.
    /// Once the number of running threads becomes zero, it wakes up the scope thread if it's waiting.
    /// The running thread count hitting zero "happens before" the scope thread returns from waiting.
    ///
    /// SAFETY:
    /// Caller must ensure that there was a matching call to increment_num_running_threadS() prior.
    pub(super) unsafe fn decrement_num_running_threads(data: NonNull<Self>, panicked: bool) {
        unsafe {
            if panicked {
                data.as_ref().thread_panicked.store(true, Ordering::Relaxed);
            }

            // Decrement the running count with a Release barrier.
            // This ensures that all data accesses and side effects before the decrement
            // "happen before" the scope thread observes the running count to be zero.
            let state_ptr = data.as_ref().sync_state.as_mut_ptr();
            let state = atomic_xsub_rel(state_ptr, ONE_RUNNING);

            let running_threads = state / ONE_RUNNING;
            assert_ne!(
                running_threads, 0,
                "decrement_num_running_threads called when not incremented"
            );

            // Wake up the scope thread if it's waiting and if we're the last running thread.
            if state == (ONE_RUNNING | WAITING_BIT) {
                // Acquire barrier ensures that both the scope_thread store and WAITING_BIT set,
                // along with the data accesses and decrements from previous threads,
                // "happen before" we start to wake up the scope thread.
                data.as_ref().fence_acquire_sync_state();

                let scope_thread = {
                    let thread_ref = &mut *data.as_ref().scope_thread.get();
                    thread_ref.take().expect("ScopeData has no thread even when WAITING_BIT is set")
                };

                // Wake up the scope thread by removing the WAITING_BIT and unparking the thread.
                // Release barrier ensures the consume of `scope_thread` "happens before" the
                // waiting scope thread observes 0 and returns to invalidate our data pointer.
                atomic_store_rel(state_ptr, 0);
                scope_thread.unpark();
            }
        }
    }

    /// Blocks the callers thread until all running threads have called decrement_num_running_threads().
    ///
    /// SAFETY:
    /// Caller must ensure that they're the sole scope_thread calling this function.
    /// There should also be no future calls to `increment_num_running_threads()` at this point.
    unsafe fn wait_for_running_threads(&self) {
        // Fast check to see if no threads are running.
        // Acquire barrier ensures the running thread count updates
        // and previous side effects on those threads "happen before" we observe 0 and return.
        if self.sync_state.load(Ordering::Acquire) == 0 {
            return;
        }

        // Register our Thread object to be unparked.
        unsafe {
            let thread_ref = &mut *self.scope_thread.get();
            let old_scope_thread = thread_ref.replace(current());
            assert!(old_scope_thread.is_none(), "multiple threads waiting on same ScopeData");
        }

        // Set the WAITING_BIT on the state to indicate there's a waiter.
        // Uses `fetch_add` over `fetch_or` as the former compiles to accelerated instructions on modern CPUs.
        // Release barrier ensures Thread registration above "happens before" WAITING_BIT is observed by last running thread.
        let state = self.sync_state.fetch_add(WAITING_BIT, Ordering::Release);
        assert_eq!(state & WAITING_BIT, 0, "multiple threads waiting on same ScopeData");

        // Don't wait if all running threads completed while we were trying to set the WAITING_BIT.
        // Acquire barrier ensures all running thread count updates and related side effects "happen before" we return.
        if state / ONE_RUNNING == 0 {
            self.fence_acquire_sync_state();
            return;
        }

        // Block the thread until the last running thread sees the WAITING_BIT and resets the state to zero.
        // Acquire barrier ensures all running thread count updates and related side effects "happen before" we return.
        loop {
            park();
            if self.sync_state.load(Ordering::Acquire) == 0 {
                return;
            }
        }
    }
}

/// Create a scope for spawning scoped threads.
///
/// The function passed to `scope` will be provided a [`Scope`] object,
/// through which scoped threads can be [spawned][`Scope::spawn`].
///
/// Unlike non-scoped threads, scoped threads can borrow non-`'static` data,
/// as the scope guarantees all threads will be joined at the end of the scope.
///
/// All threads spawned within the scope that haven't been manually joined
/// will be automatically joined before this function returns.
///
/// # Panics
///
/// If any of the automatically joined threads panicked, this function will panic.
///
/// If you want to handle panics from spawned threads,
/// [`join`][ScopedJoinHandle::join] them before the end of the scope.
///
/// # Example
///
/// ```
/// use std::thread;
///
/// let mut a = vec![1, 2, 3];
/// let mut x = 0;
///
/// thread::scope(|s| {
///     s.spawn(|| {
///         println!("hello from the first scoped thread");
///         // We can borrow `a` here.
///         dbg!(&a);
///     });
///     s.spawn(|| {
///         println!("hello from the second scoped thread");
///         // We can even mutably borrow `x` here,
///         // because no other threads are using it.
///         x += a[0] + a[2];
///     });
///     println!("hello from the main thread");
/// });
///
/// // After the scope, we can modify and access our variables again:
/// a.push(4);
/// assert_eq!(x, a.len());
/// ```
///
/// # Lifetimes
///
/// Scoped threads involve two lifetimes: `'scope` and `'env`.
///
/// The `'scope` lifetime represents the lifetime of the scope itself.
/// That is: the time during which new scoped threads may be spawned,
/// and also the time during which they might still be running.
/// Once this lifetime ends, all scoped threads are joined.
/// This lifetime starts within the `scope` function, before `f` (the argument to `scope`) starts.
/// It ends after `f` returns and all scoped threads have been joined, but before `scope` returns.
///
/// The `'env` lifetime represents the lifetime of whatever is borrowed by the scoped threads.
/// This lifetime must outlast the call to `scope`, and thus cannot be smaller than `'scope`.
/// It can be as small as the call to `scope`, meaning that anything that outlives this call,
/// such as local variables defined right before the scope, can be borrowed by the scoped threads.
///
/// The `'env: 'scope` bound is part of the definition of the `Scope` type.
#[track_caller]
#[stable(feature = "scoped_threads", since = "1.63.0")]
pub fn scope<'env, F, T>(f: F) -> T
where
    F: for<'scope> FnOnce(&'scope Scope<'scope, 'env>) -> T,
{
    // We can store the ScopeData on the stack as we're careful about accessing it intrusively.
    let data = ScopeData::default();

    // Make sure the store the ScopeData as Pinned to document in the type system
    // that it must remain valid until it is dropped at the end of this function.
    // SAFETY: the ScopeData is stored on the stack.
    let scope =
        Scope { data: unsafe { Pin::new_unchecked(&data) }, env: PhantomData, scope: PhantomData };

    // Run `f`, but catch panics so we can make sure to wait for all the threads to join.
    let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));

    // Wait until all the threads are finished.
    // SAFETY: this is the only thread that calls ScopeData::wait_for_running_threads().
    unsafe { scope.data.wait_for_running_threads() };

    // Throw any panic from `f`, or the return value of `f` if no thread panicked.
    match result {
        Err(e) => resume_unwind(e),
        Ok(_) if scope.data.thread_panicked.load(Ordering::Relaxed) => {
            panic!("a scoped thread panicked")
        }
        Ok(result) => result,
    }
}

impl<'scope, 'env> Scope<'scope, 'env> {
    /// Spawns a new thread within a scope, returning a [`ScopedJoinHandle`] for it.
    ///
    /// Unlike non-scoped threads, threads spawned with this function may
    /// borrow non-`'static` data from the outside the scope. See [`scope`] for
    /// details.
    ///
    /// The join handle provides a [`join`] method that can be used to join the spawned
    /// thread. If the spawned thread panics, [`join`] will return an [`Err`] containing
    /// the panic payload.
    ///
    /// If the join handle is dropped, the spawned thread will implicitly joined at the
    /// end of the scope. In that case, if the spawned thread panics, [`scope`] will
    /// panic after all threads are joined.
    ///
    /// This call will create a thread using default parameters of [`Builder`].
    /// If you want to specify the stack size or the name of the thread, use
    /// [`Builder::spawn_scoped`] instead.
    ///
    /// # Panics
    ///
    /// Panics if the OS fails to create a thread; use [`Builder::spawn_scoped`]
    /// to recover from such errors.
    ///
    /// [`join`]: ScopedJoinHandle::join
    #[stable(feature = "scoped_threads", since = "1.63.0")]
    pub fn spawn<F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        Builder::new().spawn_scoped(self, f).expect("failed to spawn thread")
    }
}

impl Builder {
    /// Spawns a new scoped thread using the settings set through this `Builder`.
    ///
    /// Unlike [`Scope::spawn`], this method yields an [`io::Result`] to
    /// capture any failure to create the thread at the OS level.
    ///
    /// [`io::Result`]: crate::io::Result
    ///
    /// # Panics
    ///
    /// Panics if a thread name was set and it contained null bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use std::thread;
    ///
    /// let mut a = vec![1, 2, 3];
    /// let mut x = 0;
    ///
    /// thread::scope(|s| {
    ///     thread::Builder::new()
    ///         .name("first".to_string())
    ///         .spawn_scoped(s, ||
    ///     {
    ///         println!("hello from the {:?} scoped thread", thread::current().name());
    ///         // We can borrow `a` here.
    ///         dbg!(&a);
    ///     })
    ///     .unwrap();
    ///     thread::Builder::new()
    ///         .name("second".to_string())
    ///         .spawn_scoped(s, ||
    ///     {
    ///         println!("hello from the {:?} scoped thread", thread::current().name());
    ///         // We can even mutably borrow `x` here,
    ///         // because no other threads are using it.
    ///         x += a[0] + a[2];
    ///     })
    ///     .unwrap();
    ///     println!("hello from the main thread");
    /// });
    ///
    /// // After the scope, we can modify and access our variables again:
    /// a.push(4);
    /// assert_eq!(x, a.len());
    /// ```
    #[stable(feature = "scoped_threads", since = "1.63.0")]
    pub fn spawn_scoped<'scope, 'env, F, T>(
        self,
        scope: &'scope Scope<'scope, 'env>,
        f: F,
    ) -> io::Result<ScopedJoinHandle<'scope, T>>
    where
        F: FnOnce() -> T + Send + 'scope,
        T: Send + 'scope,
    {
        Ok(ScopedJoinHandle(unsafe { self.spawn_unchecked_(f, Some(scope.data)) }?))
    }
}

impl<'scope, T> ScopedJoinHandle<'scope, T> {
    /// Extracts a handle to the underlying thread.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// thread::scope(|s| {
    ///     let t = s.spawn(|| {
    ///         println!("hello");
    ///     });
    ///     println!("thread id: {:?}", t.thread().id());
    /// });
    /// ```
    #[must_use]
    #[stable(feature = "scoped_threads", since = "1.63.0")]
    pub fn thread(&self) -> &Thread {
        &self.0.thread
    }

    /// Waits for the associated thread to finish.
    ///
    /// This function will return immediately if the associated thread has already finished.
    ///
    /// In terms of [atomic memory orderings], the completion of the associated
    /// thread synchronizes with this function returning.
    /// In other words, all operations performed by that thread
    /// [happen before](https://doc.rust-lang.org/nomicon/atomics.html#data-accesses)
    /// all operations that happen after `join` returns.
    ///
    /// If the associated thread panics, [`Err`] is returned with the panic payload.
    ///
    /// [atomic memory orderings]: crate::sync::atomic
    ///
    /// # Examples
    ///
    /// ```
    /// use std::thread;
    ///
    /// thread::scope(|s| {
    ///     let t = s.spawn(|| {
    ///         panic!("oh no");
    ///     });
    ///     assert!(t.join().is_err());
    /// });
    /// ```
    #[stable(feature = "scoped_threads", since = "1.63.0")]
    pub fn join(self) -> Result<T> {
        self.0.join()
    }

    /// Checks if the associated thread has finished running its main function.
    ///
    /// `is_finished` supports implementing a non-blocking join operation, by checking
    /// `is_finished`, and calling `join` if it returns `false`. This function does not block. To
    /// block while waiting on the thread to finish, use [`join`][Self::join].
    ///
    /// This might return `true` for a brief moment after the thread's main
    /// function has returned, but before the thread itself has stopped running.
    /// However, once this returns `true`, [`join`][Self::join] can be expected
    /// to return quickly, without blocking for any significant amount of time.
    #[stable(feature = "scoped_threads", since = "1.63.0")]
    pub fn is_finished(&self) -> bool {
        Arc::strong_count(&self.0.packet) == 1
    }
}

#[stable(feature = "scoped_threads", since = "1.63.0")]
impl fmt::Debug for Scope<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.data.sync_state.load(Ordering::Relaxed);
        let num_running_threads = state / ONE_RUNNING;
        let main_thread_waiting = state & WAITING_BIT != 0;

        f.debug_struct("Scope")
            .field("num_running_threads", &num_running_threads)
            .field("thread_panicked", &self.data.thread_panicked.load(Ordering::Relaxed))
            .field("main_thread_waiting", &main_thread_waiting)
            .finish_non_exhaustive()
    }
}

#[stable(feature = "scoped_threads", since = "1.63.0")]
impl<'scope, T> fmt::Debug for ScopedJoinHandle<'scope, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopedJoinHandle").finish_non_exhaustive()
    }
}
