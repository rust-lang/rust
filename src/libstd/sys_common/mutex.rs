use crate::sys::mutex as imp;

/// An OS-based mutual exclusion lock.
///
/// This is the thinnest cross-platform wrapper around OS mutexes. All usage of
/// this mutex is unsafe and it is recommended to instead use the safe wrapper
/// at the top level of the crate instead of this type.
pub struct Mutex(imp::Mutex);

unsafe impl Sync for Mutex {}

impl Mutex {
    /// Creates a new mutex for use.
    ///
    /// Behavior is undefined if the mutex is moved after it is
    /// first used with any of the functions below.
    /// Also, until `init` is called, behavior is undefined if this
    /// mutex is ever used reentrantly, i.e., `raw_lock` or `try_lock`
    /// are called by the thread currently holding the lock.
    pub const fn new() -> Mutex { Mutex(imp::Mutex::new()) }

    /// Prepare the mutex for use.
    ///
    /// This should be called once the mutex is at a stable memory address.
    /// If called, this must be the very first thing that happens to the mutex.
    /// Calling it in parallel with or after any operation (including another
    /// `init()`) is undefined behavior.
    #[inline]
    pub unsafe fn init(&mut self) { self.0.init() }

    /// Locks the mutex blocking the current thread until it is available.
    ///
    /// Behavior is undefined if the mutex has been moved between this and any
    /// previous function call.
    #[inline]
    pub unsafe fn raw_lock(&self) { self.0.lock() }

    /// Calls raw_lock() and then returns an RAII guard to guarantee the mutex
    /// will be unlocked.
    #[inline]
    pub unsafe fn lock(&self) -> MutexGuard<'_> {
        self.raw_lock();
        MutexGuard(&self.0)
    }

    /// Attempts to lock the mutex without blocking, returning whether it was
    /// successfully acquired or not.
    ///
    /// Behavior is undefined if the mutex has been moved between this and any
    /// previous function call.
    #[inline]
    pub unsafe fn try_lock(&self) -> bool { self.0.try_lock() }

    /// Unlocks the mutex.
    ///
    /// Behavior is undefined if the current thread does not actually hold the
    /// mutex.
    ///
    /// Consider switching from the pair of raw_lock() and raw_unlock() to
    /// lock() whenever possible.
    #[inline]
    pub unsafe fn raw_unlock(&self) { self.0.unlock() }

    /// Deallocates all resources associated with this mutex.
    ///
    /// Behavior is undefined if there are current or will be future users of
    /// this mutex.
    #[inline]
    pub unsafe fn destroy(&self) { self.0.destroy() }
}

// not meant to be exported to the outside world, just the containing module
pub fn raw(mutex: &Mutex) -> &imp::Mutex { &mutex.0 }

#[must_use]
/// A simple RAII utility for the above Mutex without the poisoning semantics.
pub struct MutexGuard<'a>(&'a imp::Mutex);

impl Drop for MutexGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        unsafe { self.0.unlock(); }
    }
}
