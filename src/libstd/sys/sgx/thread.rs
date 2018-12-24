// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use boxed::FnBox;
use ffi::CStr;
use io;
use time::Duration;

use super::abi::usercalls;

pub struct Thread(task_queue::JoinHandle);

pub const DEFAULT_MIN_STACK_SIZE: usize = 4096;

mod task_queue {
    use sync::{Mutex, MutexGuard, Once};
    use sync::mpsc;
    use boxed::FnBox;

    pub type JoinHandle = mpsc::Receiver<()>;

    pub(super) struct Task {
        p: Box<dyn FnBox()>,
        done: mpsc::Sender<()>,
    }

    impl Task {
        pub(super) fn new(p: Box<dyn FnBox()>) -> (Task, JoinHandle) {
            let (done, recv) = mpsc::channel();
            (Task { p, done }, recv)
        }

        pub(super) fn run(self) {
            (self.p)();
            let _ = self.done.send(());
        }
    }

    static TASK_QUEUE_INIT: Once = Once::new();
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
    pub unsafe fn new(_stack: usize, p: Box<dyn FnBox()>)
        -> io::Result<Thread>
    {
        let mut queue_lock = task_queue::lock();
        usercalls::launch_thread()?;
        let (task, handle) = task_queue::Task::new(p);
        queue_lock.push(task);
        Ok(Thread(handle))
    }

    pub(super) fn entry() {
        let mut guard = task_queue::lock();
        let task = guard.pop().expect("Thread started but no tasks pending");
        drop(guard); // make sure to not hold the task queue lock longer than necessary
        task.run()
    }

    pub fn yield_now() {
        assert_eq!(
            usercalls::wait(0, usercalls::raw::WAIT_NO).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );
    }

    pub fn set_name(_name: &CStr) {
        // FIXME: could store this pointer in TLS somewhere
    }

    pub fn sleep(_dur: Duration) {
        panic!("can't sleep"); // FIXME
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
