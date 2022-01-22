use super::{current, park, Builder, JoinInner, Result, Thread};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use crate::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crate::sync::Arc;

/// A scope to spawn scoped threads in.
///
/// See [`scope`] for details.
pub struct Scope<'env> {
    data: ScopeData,
    /// Invariance over 'env, to make sure 'env cannot shrink,
    /// which is necessary for soundness.
    ///
    /// Without invariance, this would compile fine but be unsound:
    ///
    /// ```compile_fail
    /// #![feature(scoped_threads)]
    ///
    /// std::thread::scope(|s| {
    ///     s.spawn(|s| {
    ///         let a = String::from("abcd");
    ///         s.spawn(|_| println!("{:?}", a)); // might run after `a` is dropped
    ///     });
    /// });
    /// ```
    env: PhantomData<&'env mut &'env ()>,
}

/// An owned permission to join on a scoped thread (block on its termination).
///
/// See [`Scope::spawn`] for details.
pub struct ScopedJoinHandle<'scope, T>(JoinInner<'scope, T>);

pub(super) struct ScopeData {
    num_running_threads: AtomicUsize,
    a_thread_panicked: AtomicBool,
    main_thread: Thread,
}

impl ScopeData {
    pub(super) fn increment_num_running_threads(&self) {
        // We check for 'overflow' with usize::MAX / 2, to make sure there's no
        // chance it overflows to 0, which would result in unsoundness.
        if self.num_running_threads.fetch_add(1, Ordering::Relaxed) > usize::MAX / 2 {
            // This can only reasonably happen by mem::forget()'ing many many ScopedJoinHandles.
            self.decrement_num_running_threads(false);
            panic!("too many running threads in thread scope");
        }
    }
    pub(super) fn decrement_num_running_threads(&self, panic: bool) {
        if panic {
            self.a_thread_panicked.store(true, Ordering::Relaxed);
        }
        if self.num_running_threads.fetch_sub(1, Ordering::Release) == 1 {
            self.main_thread.unpark();
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
/// #![feature(scoped_threads)]
/// use std::thread;
///
/// let mut a = vec![1, 2, 3];
/// let mut x = 0;
///
/// thread::scope(|s| {
///     s.spawn(|_| {
///         println!("hello from the first scoped thread");
///         // We can borrow `a` here.
///         dbg!(&a);
///     });
///     s.spawn(|_| {
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
#[track_caller]
pub fn scope<'env, F, T>(f: F) -> T
where
    F: FnOnce(&Scope<'env>) -> T,
{
    let scope = Scope {
        data: ScopeData {
            num_running_threads: AtomicUsize::new(0),
            main_thread: current(),
            a_thread_panicked: AtomicBool::new(false),
        },
        env: PhantomData,
    };

    // Run `f`, but catch panics so we can make sure to wait for all the threads to join.
    let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));

    // Wait until all the threads are finished.
    while scope.data.num_running_threads.load(Ordering::Acquire) != 0 {
        park();
    }

    // Throw any panic from `f`, or the return value of `f` if no thread panicked.
    match result {
        Err(e) => resume_unwind(e),
        Ok(_) if scope.data.a_thread_panicked.load(Ordering::Relaxed) => {
            panic!("a scoped thread panicked")
        }
        Ok(result) => result,
    }
}

impl<'env> Scope<'env> {
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
    pub fn spawn<'scope, F, T>(&'scope self, f: F) -> ScopedJoinHandle<'scope, T>
    where
        F: FnOnce(&Scope<'env>) -> T + Send + 'env,
        T: Send + 'env,
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
    /// #![feature(scoped_threads)]
    /// use std::thread;
    ///
    /// let mut a = vec![1, 2, 3];
    /// let mut x = 0;
    ///
    /// thread::scope(|s| {
    ///     thread::Builder::new()
    ///         .name("first".to_string())
    ///         .spawn_scoped(s, |_|
    ///     {
    ///         println!("hello from the {:?} scoped thread", thread::current().name());
    ///         // We can borrow `a` here.
    ///         dbg!(&a);
    ///     })
    ///     .unwrap();
    ///     thread::Builder::new()
    ///         .name("second".to_string())
    ///         .spawn_scoped(s, |_|
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
    pub fn spawn_scoped<'scope, 'env, F, T>(
        self,
        scope: &'scope Scope<'env>,
        f: F,
    ) -> io::Result<ScopedJoinHandle<'scope, T>>
    where
        F: FnOnce(&Scope<'env>) -> T + Send + 'env,
        T: Send + 'env,
    {
        Ok(ScopedJoinHandle(unsafe { self.spawn_unchecked_(|| f(scope), Some(&scope.data)) }?))
    }
}

impl<'scope, T> ScopedJoinHandle<'scope, T> {
    /// Extracts a handle to the underlying thread.
    ///
    /// # Examples
    ///
    /// ```
    /// #![feature(scoped_threads)]
    /// #![feature(thread_is_running)]
    ///
    /// use std::thread;
    ///
    /// thread::scope(|s| {
    ///     let t = s.spawn(|_| {
    ///         println!("hello");
    ///     });
    ///     println!("thread id: {:?}", t.thread().id());
    /// });
    /// ```
    #[must_use]
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
    /// #![feature(scoped_threads)]
    /// #![feature(thread_is_running)]
    ///
    /// use std::thread;
    ///
    /// thread::scope(|s| {
    ///     let t = s.spawn(|_| {
    ///         panic!("oh no");
    ///     });
    ///     assert!(t.join().is_err());
    /// });
    /// ```
    pub fn join(self) -> Result<T> {
        self.0.join()
    }

    /// Checks if the associated thread is still running its main function.
    ///
    /// This might return `false` for a brief moment after the thread's main
    /// function has returned, but before the thread itself has stopped running.
    #[unstable(feature = "thread_is_running", issue = "90470")]
    pub fn is_running(&self) -> bool {
        Arc::strong_count(&self.0.packet) > 1
    }
}

impl<'env> fmt::Debug for Scope<'env> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scope")
            .field("num_running_threads", &self.data.num_running_threads.load(Ordering::Relaxed))
            .field("a_thread_panicked", &self.data.a_thread_panicked.load(Ordering::Relaxed))
            .field("main_thread", &self.data.main_thread)
            .finish_non_exhaustive()
    }
}

impl<'scope, T> fmt::Debug for ScopedJoinHandle<'scope, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScopedJoinHandle").finish_non_exhaustive()
    }
}
