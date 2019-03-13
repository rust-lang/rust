use crate::time::Duration;
use crate::sys_common::mutex::{self, Mutex};
use crate::sys::condvar as imp;

/// An OS-based condition variable.
///
/// This structure is the lowest layer possible on top of the OS-provided
/// condition variables. It is consequently entirely unsafe to use. It is
/// recommended to use the safer types at the top level of this crate instead of
/// this type.
pub struct Condvar(imp::Condvar);

impl Condvar {
    /// Creates a new condition variable for use.
    ///
    /// Behavior is undefined if the condition variable is moved after it is
    /// first used with any of the functions below.
    pub const fn new() -> Condvar { Condvar(imp::Condvar::new()) }

    /// Prepares the condition variable for use.
    ///
    /// This should be called once the condition variable is at a stable memory
    /// address.
    #[inline]
    pub unsafe fn init(&mut self) { self.0.init() }

    /// Signals one waiter on this condition variable to wake up.
    #[inline]
    pub unsafe fn notify_one(&self) { self.0.notify_one() }

    /// Awakens all current waiters on this condition variable.
    #[inline]
    pub unsafe fn notify_all(&self) { self.0.notify_all() }

    /// Waits for a signal on the specified mutex.
    ///
    /// Behavior is undefined if the mutex is not locked by the current thread.
    /// Behavior is also undefined if more than one mutex is used concurrently
    /// on this condition variable.
    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) { self.0.wait(mutex::raw(mutex)) }

    /// Waits for a signal on the specified mutex with a timeout duration
    /// specified by `dur` (a relative time into the future).
    ///
    /// Behavior is undefined if the mutex is not locked by the current thread.
    /// Behavior is also undefined if more than one mutex is used concurrently
    /// on this condition variable.
    #[inline]
    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        self.0.wait_timeout(mutex::raw(mutex), dur)
    }

    /// Deallocates all resources associated with this condition variable.
    ///
    /// Behavior is undefined if there are current or will be future users of
    /// this condition variable.
    #[inline]
    pub unsafe fn destroy(&self) { self.0.destroy() }
}
