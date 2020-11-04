#![cfg_attr(test, allow(dead_code))] // why is this necessary?
use super::unsupported;
use crate::ffi::CStr;
use crate::io;
use crate::num::NonZeroUsize;
use crate::time::Duration;

use super::abi::thread;
use super::abi::usercalls;

pub struct Thread(task_queue::JoinHandle);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

pub use self::task_queue::JoinNotifier;

mod tcs_queue {
    use super::super::abi::mem as sgx_mem;
    use super::super::abi::thread;
    use crate::ptr::NonNull;
    use crate::sync::{Mutex, MutexGuard, Once};

    #[derive(Clone, PartialEq, Eq, Debug)]
    pub(super) struct Tcs {
        address: NonNull<u8>,
    }

    impl Tcs {
        fn new(address: NonNull<u8>) -> Tcs {
            Tcs { address }
        }

        pub(super) fn address(&self) -> &NonNull<u8> {
            &self.address
        }
    }

    /// A queue of not running TCS structs
    static mut TCS_QUEUE: Option<Mutex<Vec<Tcs>>> = None;
    static TCS_QUEUE_INIT: Once = Once::new();

    fn init_tcs_queue() -> Vec<Tcs> {
        sgx_mem::static_tcses()
            .iter()
            .filter_map(|addr| if NonNull::new(*addr as _) != Some(thread::current()) {
                    Some(Tcs::new(NonNull::new(*addr as _).expect("Compile-time value unexpected NULL")))
                } else {
                    None
                })
            .collect()
    }

    fn lock() -> MutexGuard<'static, Vec<Tcs>> {
        unsafe {
            TCS_QUEUE_INIT.call_once(|| TCS_QUEUE = Some(Mutex::new(init_tcs_queue())));
            TCS_QUEUE.as_ref().unwrap().lock().unwrap()
        }
    }

    pub(super) fn take_tcs() -> Option<Tcs> {
        let mut tcs_queue = lock();
        if let Some(tcs) = tcs_queue.pop() { Some(tcs) } else { None }
    }

    pub(super) fn add_tcs(tcs: Tcs) {
        let mut tcs_queue = lock();
        tcs_queue.insert(0, tcs);
    }
}

mod task_queue {
    use super::tcs_queue::{self, Tcs};
    use super::wait_notify;
    use crate::ptr::NonNull;
    use crate::sync::mpsc;
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
        tcs: Tcs,
    }

    impl Task {
        pub(super) fn new(tcs: Tcs, p: Box<dyn FnOnce()>) -> (Task, JoinHandle) {
            let (done, recv) = wait_notify::new();
            let done = JoinNotifier(Some(done));
            let task = Task { p, done, tcs };
            (task, recv)
        }

        pub(super) fn run(self) -> JoinNotifier {
            let Task { tcs, p, done } = self;

            p();

            tcs_queue::add_tcs(tcs);
            done
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

    pub(super) fn take_task(tcs: NonNull<u8>) -> Option<Task> {
        let mut tasks = lock();
        let (i, _) = tasks.iter().enumerate().find(|(_i, task)| *task.tcs.address() == tcs)?;
        let task = tasks.remove(i);
        Some(task)
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
        let tcs = tcs_queue::take_tcs().ok_or(io::Error::from(io::ErrorKind::WouldBlock))?;
        let mut tasks = task_queue::lock();
        unsafe { usercalls::launch_thread(Some(*tcs.address()))? };
        let (task, handle) = task_queue::Task::new(tcs, p);
        tasks.push(task);
        Ok(Thread(handle))
    }

    pub(super) fn entry() -> JoinNotifier {
        let task = task_queue::take_task(thread::current())
            .expect("enclave entered through TCS unexpectedly");
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
