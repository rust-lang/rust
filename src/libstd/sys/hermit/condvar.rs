use crate::cmp;
use crate::sys::mutex::Mutex;
use crate::time::Duration;

pub struct Condvar {
    identifier: usize,
}

extern "C" {
   fn sys_notify(id: usize, count: i32) -> i32;
   fn sys_add_queue(id: usize, timeout_ns: i64) -> i32;
   fn sys_wait(id: usize) -> i32;
   fn sys_destroy_queue(id: usize) -> i32;
}

impl Condvar {
    pub const fn new() -> Condvar {
        Condvar { identifier: 0 }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        // nothing to do
    }

    pub unsafe fn notify_one(&self) {
         let _ = sys_notify(self.id(), 1);
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
         let _ = sys_notify(self.id(), -1 /* =all */);
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        // add current task to the wait queue
        let _ = sys_add_queue(self.id(), -1 /* no timeout */);
        mutex.unlock();
        let _ = sys_wait(self.id());
        mutex.lock();
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let nanos = dur.as_nanos();
        let nanos = cmp::min(i64::max_value() as u128, nanos);

        // add current task to the wait queue
        let _ = sys_add_queue(self.id(), nanos as i64);

        mutex.unlock();
        // If the return value is !0 then a timeout happened, so we return
        // `false` as we weren't actually notified.
        let ret = sys_wait(self.id()) == 0;
        mutex.lock();

        ret
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let _ = sys_destroy_queue(self.id());
    }

    #[inline]
    fn id(&self) -> usize {
        &self.identifier as *const usize as usize
    }
}
