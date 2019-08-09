#![cfg_attr(test, allow(dead_code))] // why is this necessary?
use crate::ffi::CStr;
use crate::io;
use crate::time::Duration;

use super::abi::usercalls;

pub struct Thread(task_queue::JoinHandle);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

mod task_queue {
    use crate::sync::{Mutex, MutexGuard, Once};
    use crate::sync::mpsc;

    pub type JoinHandle = mpsc::Receiver<()>;

    pub(super) struct Task {
        p: Box<dyn FnOnce()>,
        done: mpsc::Sender<()>,
    }

    impl Task {
        pub(super) fn new(p: Box<dyn FnOnce()>) -> (Task, JoinHandle) {
            let (done, recv) = mpsc::channel();
            (Task { p, done }, recv)
        }

        pub(super) fn run(self) {
            (self.p)();
            let _ = self.done.send(());
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
            TASK_QUEUE_INIT.call_once(|| TASK_QUEUE = Some(Default::default()) );
            TASK_QUEUE.as_ref().unwrap().lock().unwrap()
        }
    }
}

impl Thread {
    // unsafe: see thread::Builder::spawn_unchecked for safety requirements
    pub unsafe fn new(_stack: usize, p: Box<dyn FnOnce()>)
        -> io::Result<Thread>
    {
        let mut queue_lock = task_queue::lock();
        usercalls::launch_thread()?;
        let (task, handle) = task_queue::Task::new(p);
        queue_lock.push(task);
        Ok(Thread(handle))
    }

    pub(super) fn entry() {
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

    pub fn sleep(_dur: Duration) {
        rtabort!("can't sleep"); // FIXME
    }

    pub fn join(self) {
        let _ = self.0.recv();
    }
}

pub mod guard {
    pub type Guard = !;
    pub unsafe fn current() -> Option<Guard> { None }
    pub unsafe fn init() -> Option<Guard> { None }
}
