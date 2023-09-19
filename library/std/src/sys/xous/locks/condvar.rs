use super::mutex::Mutex;
use crate::os::xous::ffi::{blocking_scalar, scalar};
use crate::os::xous::services::ticktimer_server;
use crate::sync::Mutex as StdMutex;
use crate::time::Duration;

// The implementation is inspired by Andrew D. Birrell's paper
// "Implementing Condition Variables with Semaphores"

pub struct Condvar {
    counter: StdMutex<usize>,
}

unsafe impl Send for Condvar {}
unsafe impl Sync for Condvar {}

impl Condvar {
    #[inline]
    #[rustc_const_stable(feature = "const_locks", since = "1.63.0")]
    pub const fn new() -> Condvar {
        Condvar { counter: StdMutex::new(0) }
    }

    pub fn notify_one(&self) {
        let mut counter = self.counter.lock().unwrap();
        if *counter <= 0 {
            return;
        } else {
            *counter -= 1;
        }
        let result = blocking_scalar(
            ticktimer_server(),
            crate::os::xous::services::TicktimerScalar::NotifyCondition(self.index(), 1).into(),
        );
        drop(counter);
        result.expect("failure to send NotifyCondition command");
    }

    pub fn notify_all(&self) {
        let mut counter = self.counter.lock().unwrap();
        if *counter <= 0 {
            return;
        }
        let result = blocking_scalar(
            ticktimer_server(),
            crate::os::xous::services::TicktimerScalar::NotifyCondition(self.index(), *counter)
                .into(),
        );
        *counter = 0;
        drop(counter);

        result.expect("failure to send NotifyCondition command");
    }

    fn index(&self) -> usize {
        self as *const Condvar as usize
    }

    pub unsafe fn wait(&self, mutex: &Mutex) {
        let mut counter = self.counter.lock().unwrap();
        *counter += 1;
        unsafe { mutex.unlock() };
        drop(counter);

        let result = blocking_scalar(
            ticktimer_server(),
            crate::os::xous::services::TicktimerScalar::WaitForCondition(self.index(), 0).into(),
        );
        unsafe { mutex.lock() };

        result.expect("Ticktimer: failure to send WaitForCondition command");
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let mut counter = self.counter.lock().unwrap();
        *counter += 1;
        unsafe { mutex.unlock() };
        drop(counter);

        let mut millis = dur.as_millis() as usize;
        if millis == 0 {
            millis = 1;
        }

        let result = blocking_scalar(
            ticktimer_server(),
            crate::os::xous::services::TicktimerScalar::WaitForCondition(self.index(), millis)
                .into(),
        );
        unsafe { mutex.lock() };

        let result = result.expect("Ticktimer: failure to send WaitForCondition command")[0] == 0;

        // If we awoke due to a timeout, decrement the wake count, as that would not have
        // been done in the `notify()` call.
        if !result {
            *self.counter.lock().unwrap() -= 1;
        }
        result
    }
}

impl Drop for Condvar {
    fn drop(&mut self) {
        scalar(
            ticktimer_server(),
            crate::os::xous::services::TicktimerScalar::FreeCondition(self.index()).into(),
        )
        .ok();
    }
}
