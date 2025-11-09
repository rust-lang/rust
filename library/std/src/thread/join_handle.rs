use super::Result;
use super::lifecycle::JoinInner;
use super::thread::Thread;
use crate::fmt;
use crate::sys::thread as imp;
use crate::sys_common::{AsInner, IntoInner};

/// An owned permission to join on a thread (block on its termination).
///
/// A `JoinHandle` *detaches* the associated thread when it is dropped, which
/// means that there is no longer any handle to the thread and no way to `join`
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
/// A thread being detached and outliving the thread that spawned it:
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
/// [`thread::Builder::spawn`]: super::Builder::spawn
/// [`thread::spawn`]: super::spawn
#[stable(feature = "rust1", since = "1.0.0")]
#[cfg_attr(target_os = "teeos", must_use)]
pub struct JoinHandle<T>(pub(super) JoinInner<'static, T>);

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
    #[must_use]
    pub fn thread(&self) -> &Thread {
        self.0.thread()
    }

    /// Waits for the associated thread to finish.
    ///
    /// This function will return immediately if the associated thread has already finished.
    ///
    /// In terms of [atomic memory orderings],  the completion of the associated
    /// thread synchronizes with this function returning. In other words, all
    /// operations performed by that thread [happen
    /// before](https://doc.rust-lang.org/nomicon/atomics.html#data-accesses) all
    /// operations that happen after `join` returns.
    ///
    /// If the associated thread panics, [`Err`] is returned with the parameter given
    /// to [`panic!`] (though see the Notes below).
    ///
    /// [`Err`]: crate::result::Result::Err
    /// [atomic memory orderings]: crate::sync::atomic
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
    ///
    /// # Notes
    ///
    /// If a "foreign" unwinding operation (e.g. an exception thrown from C++
    /// code, or a `panic!` in Rust code compiled or linked with a different
    /// runtime) unwinds all the way to the thread root, the process may be
    /// aborted; see the Notes on [`thread::spawn`]. If the process is not
    /// aborted, this function will return a `Result::Err` containing an opaque
    /// type.
    ///
    /// [`catch_unwind`]: ../../std/panic/fn.catch_unwind.html
    /// [`thread::spawn`]: super::spawn
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn join(self) -> Result<T> {
        self.0.join()
    }

    /// Checks if the associated thread has finished running its main function.
    ///
    /// `is_finished` supports implementing a non-blocking join operation, by checking
    /// `is_finished`, and calling `join` if it returns `true`. This function does not block. To
    /// block while waiting on the thread to finish, use [`join`][Self::join].
    ///
    /// This might return `true` for a brief moment after the thread's main
    /// function has returned, but before the thread itself has stopped running.
    /// However, once this returns `true`, [`join`][Self::join] can be expected
    /// to return quickly, without blocking for any significant amount of time.
    #[stable(feature = "thread_is_running", since = "1.61.0")]
    pub fn is_finished(&self) -> bool {
        self.0.is_finished()
    }
}

impl<T> AsInner<imp::Thread> for JoinHandle<T> {
    fn as_inner(&self) -> &imp::Thread {
        self.0.as_inner()
    }
}

impl<T> IntoInner<imp::Thread> for JoinHandle<T> {
    fn into_inner(self) -> imp::Thread {
        self.0.into_inner()
    }
}

#[stable(feature = "std_debug", since = "1.16.0")]
impl<T> fmt::Debug for JoinHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JoinHandle").finish_non_exhaustive()
    }
}
