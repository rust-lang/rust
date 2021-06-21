#![cfg_attr(test, allow(dead_code))] // why is this necessary?
use super::unsupported;
use crate::ffi::CStr;
use crate::io;
use crate::num::NonZeroUsize;
use crate::time::Duration;

use super::abi::usercalls;

pub struct Thread(task_queue::JoinHandle);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

pub use self::task_queue::JoinNotifier;

mod task_queue {
    use super::wait_notify;
    use crate::sync::{Mutex, MutexGuard, Once};

    pub type JoinHandle = wait_notify::Waiter;

    pub struct JoinNotifier(Option<wait_notify::Notifier>);

    impl Drop for JoinNotifier {
        fn drop(&mut self) {
            self.0.take().unwrap().notify();
        }
    }

    pub(super) struct Task {
        p: Box<dyn FnOnce()>,
        done: JoinNotifier,
    }

    impl Task {
        pub(super) fn new(p: Box<dyn FnOnce()>) -> (Task, JoinHandle) {
            let (done, recv) = wait_notify::new();
            let done = JoinNotifier(Some(done));
            (Task { p, done }, recv)
        }

        pub(super) fn run(self) -> JoinNotifier {
            (self.p)();
            self.done
        }
    }

    #[cfg_attr(test, linkage = "available_externally")]
    #[export_name = "_ZN16__rust_internals3std3sys3sgx6thread15TASK_QUEUE_INITE"]
    static TASK_QUEUE_INIT: Once = Once::new();
    #[cfg_attr(test, linkage = "available_externally")]
    #[export_name = "_ZN16__rust_internals3std3sys3sgx6thread10TASK_QUEUEE"]
    static mut TASK_QUEUE: Option<Mutex<Vec<Task>>> = None;

    pub(super) fn lock() -> MutexGuard<'static, Vec<Task>> {
        unsafe {
            TASK_QUEUE_INIT.call_once(|| TASK_QUEUE = Some(Default::default()));
            TASK_QUEUE.as_ref().unwrap().lock().unwrap()
        }
    }
}

/// This module provides a synchronization primitive that does not use thread
/// local variables. This is needed for signaling that a thread has finished
/// execution. The signal is sent once all TLS destructors have finished at
/// which point no new thread locals should be created.
pub mod wait_notify {
    use super::super::waitqueue::{SpinMutex, WaitQueue, WaitVariable};
    use crate::sync::Arc;

    pub struct Notifier(Arc<SpinMutex<WaitVariable<bool>>>);

    impl Notifier {
        /// Notify the waiter. The waiter is either notified right away (if
        /// currently blocked in `Waiter::wait()`) or later when it calls the
        /// `Waiter::wait()` method.
        pub fn notify(self) {
            let mut guard = self.0.lock();
            *guard.lock_var_mut() = true;
            let _ = WaitQueue::notify_one(guard);
        }
    }

    pub struct Waiter(Arc<SpinMutex<WaitVariable<bool>>>);

    impl Waiter {
        /// Wait for a notification. If `Notifier::notify()` has already been
        /// called, this will return immediately, otherwise the current thread
        /// is blocked until notified.
        pub fn wait(self) {
            let guard = self.0.lock();
            if *guard.lock_var() {
                return;
            }
            WaitQueue::wait(guard, || {});
        }
    }

    pub fn new() -> (Notifier, Waiter) {
        let inner = Arc::new(SpinMutex::new(WaitVariable::new(false)));
        (Notifier(inner.clone()), Waiter(inner))
    }
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(_stack: usize, p: Box<dyn FnOnce()>) -> io::Result<Thread> {
        let mut queue_lock = task_queue::lock();
        unsafe { usercalls::launch_thread()? };
        let (task, handle) = task_queue::Task::new(p);
        queue_lock.push(task);
        Ok(Thread(handle))
    }

    pub(super) fn entry() -> JoinNotifier {
        let mut pending_tasks = task_queue::lock();
        let task = rtunwrap!(Some, pending_tasks.pop());
        drop(pending_tasks); // make sure to not hold the task queue lock longer than necessary
        task.run()
    }

    pub fn yield_now() {
        let wait_error = rtunwrap!(Err, usercalls::wait(0, usercalls::raw::WAIT_NO));
        rtassert!(wait_error.kind() == io::ErrorKind::WouldBlock);
    }

    pub fn set_name(_name: &CStr) {
        // FIXME: could store this pointer in TLS somewhere
    }

    pub fn sleep(dur: Duration) {
        usercalls::wait_timeout(0, dur, || true);
    }

    pub fn join(self) {
        self.0.wait();
    }
}

pub fn available_concurrency() -> io::Result<NonZeroUsize> {
    unsupported()
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> {
        None
    }
    pub unsafe fn init() -> Option<Guard> {
        None
    }
}
