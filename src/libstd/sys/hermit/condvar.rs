use crate::cmp;
use crate::sys::hermit::abi;
use crate::sys::mutex::Mutex;
use crate::time::Duration;

pub struct Condvar {
    identifier: usize,
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
         let _ = abi::notify(self.id(), 1);
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
         let _ = abi::notify(self.id(), -1 /* =all */);
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        // add current task to the wait queue
        let _ = abi::add_queue(self.id(), -1 /* no timeout */);
        mutex.unlock();
        let _ = abi::wait(self.id());
        mutex.lock();
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let nanos = dur.as_nanos();
        let nanos = cmp::min(i64::max_value() as u128, nanos);

        // add current task to the wait queue
        let _ = abi::add_queue(self.id(), nanos as i64);

        mutex.unlock();
        // If the return value is !0 then a timeout happened, so we return
        // `false` as we weren't actually notified.
        let ret = abi::wait(self.id()) == 0;
        mutex.lock();

        ret
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        let _ = abi::destroy_queue(self.id());
    }

    #[inline]
    fn id(&self) -> usize {
        &self.identifier as *const usize as usize
    }
}
