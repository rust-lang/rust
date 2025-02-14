#![cfg_attr(test, allow(dead_code))] // why is this necessary?

use super::abi::usercalls;
use super::unsupported;
use crate::ffi::CStr;
use crate::io;
use crate::num::NonZero;
use crate::time::Duration;

pub struct Thread(task_queue::JoinHandle);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

pub use self::task_queue::JoinNotifier;

mod task_queue {
    use super::wait_notify;
    use crate::sync::{Mutex, MutexGuard};

    pub type JoinHandle = wait_notify::Waiter;

    pub struct JoinNotifier(Option<wait_notify::Notifier>);

    impl Drop for JoinNotifier {
        fn drop(&mut self) {
            self.0.take().unwrap().notify();
        }
    }

    pub(super) struct Task {
        p: Box<dyn FnOnce() + Send>,
        done: JoinNotifier,
    }

    impl Task {
        pub(super) fn new(p: Box<dyn FnOnce() + Send>) -> (Task, JoinHandle) {
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
    #[unsafe(export_name = "_ZN16__rust_internals3std3sys3sgx6thread10TASK_QUEUEE")]
    static TASK_QUEUE: Mutex<Vec<Task>> = Mutex::new(Vec::new());

    pub(super) fn lock() -> MutexGuard<'static, Vec<Task>> {
        TASK_QUEUE.lock().unwrap()
    }
}

/// This module provides a synchronization primitive that does not use thread
/// local variables. This is needed for signaling that a thread has finished
/// execution. The signal is sent once all TLS destructors have finished at
/// which point no new thread locals should be created.
pub mod wait_notify {
    use crate::pin::Pin;
    use crate::sync::Arc;
    use crate::sys::sync::Parker;

    pub struct Notifier(Arc<Parker>);

    impl Notifier {
        /// Notify the waiter. The waiter is either notified right away (if
        /// currently blocked in `Waiter::wait()`) or later when it calls the
        /// `Waiter::wait()` method.
        pub fn notify(self) {
            Pin::new(&*self.0).unpark()
        }
    }

    pub struct Waiter(Arc<Parker>);

    impl Waiter {
        /// Wait for a notification. If `Notifier::notify()` has already been
        /// called, this will return immediately, otherwise the current thread
        /// is blocked until notified.
        pub fn wait(self) {
            // SAFETY:
            // This is only ever called on one thread.
            unsafe { Pin::new(&*self.0).park() }
        }
    }

    pub fn new() -> (Notifier, Waiter) {
        let inner = Arc::new(Parker::new());
        (Notifier(inner.clone()), Waiter(inner))
    }
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(_stack: usize, p: Box<dyn FnOnce() + Send>) -> io::Result<Thread> {
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

    /// SGX should protect in-enclave data from the outside (attacker),
    /// so there should be no data leakage to the OS,
    /// and therefore also no 1-1 mapping between SGX thread names and OS thread names.
    ///
    /// This is why the method is intentionally No-Op.
    pub fn set_name(_name: &CStr) {
        // Note that the internally visible SGX thread name is already provided
        // by the platform-agnostic (target-agnostic) Rust thread code.
        // This can be observed in the [`std::thread::tests::test_named_thread`] test,
        // which succeeds as-is with the SGX target.
    }

    pub fn sleep(dur: Duration) {
        usercalls::wait_timeout(0, dur, || true);
    }

    pub fn join(self) {
        self.0.wait();
    }
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    unsupported()
}
