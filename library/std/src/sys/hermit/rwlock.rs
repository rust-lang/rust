use crate::cell::UnsafeCell;
use crate::sys::locks::{Condvar, Mutex};

pub struct RwLock {
    lock: Mutex,
    cond: Condvar,
    state: UnsafeCell<State>,
}

enum State {
    Unlocked,
    Reading(usize),
    Writing,
}

unsafe impl Send for RwLock {}
unsafe impl Sync for RwLock {}

// This rwlock implementation is a relatively simple implementation which has a
// condition variable for readers/writers as well as a mutex protecting the
// internal state of the lock. A current downside of the implementation is that
// unlocking the lock will notify *all* waiters rather than just readers or just
// writers. This can cause lots of "thundering stampede" problems. While
// hopefully correct this implementation is very likely to want to be changed in
// the future.

impl RwLock {
    pub const fn new() -> RwLock {
        RwLock { lock: Mutex::new(), cond: Condvar::new(), state: UnsafeCell::new(State::Unlocked) }
    }

    #[inline]
    pub fn read(&self) {
        unsafe {
            self.lock.lock();
            while !(*self.state.get()).inc_readers() {
                self.cond.wait(&self.lock);
            }
            self.lock.unlock();
        }
    }

    #[inline]
    pub fn try_read(&self) -> bool {
        unsafe {
            self.lock.lock();
            let ok = (*self.state.get()).inc_readers();
            self.lock.unlock();
            ok
        }
    }

    #[inline]
    pub fn write(&self) {
        unsafe {
            self.lock.lock();
            while !(*self.state.get()).inc_writers() {
                self.cond.wait(&self.lock);
            }
            self.lock.unlock();
        }
    }

    #[inline]
    pub fn try_write(&self) -> bool {
        unsafe {
            self.lock.lock();
            let ok = (*self.state.get()).inc_writers();
            self.lock.unlock();
            ok
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        self.lock.lock();
        let notify = (*self.state.get()).dec_readers();
        self.lock.unlock();
        if notify {
            // FIXME: should only wake up one of these some of the time
            self.cond.notify_all();
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        self.lock.lock();
        (*self.state.get()).dec_writers();
        self.lock.unlock();
        // FIXME: should only wake up one of these some of the time
        self.cond.notify_all();
    }
}

impl State {
    fn inc_readers(&mut self) -> bool {
        match *self {
            State::Unlocked => {
                *self = State::Reading(1);
                true
            }
            State::Reading(ref mut cnt) => {
                *cnt += 1;
                true
            }
            State::Writing => false,
        }
    }

    fn inc_writers(&mut self) -> bool {
        match *self {
            State::Unlocked => {
                *self = State::Writing;
                true
            }
            State::Reading(_) | State::Writing => false,
        }
    }

    fn dec_readers(&mut self) -> bool {
        let zero = match *self {
            State::Reading(ref mut cnt) => {
                *cnt -= 1;
                *cnt == 0
            }
            State::Unlocked | State::Writing => invalid(),
        };
        if zero {
            *self = State::Unlocked;
        }
        zero
    }

    fn dec_writers(&mut self) {
        match *self {
            State::Writing => {}
            State::Unlocked | State::Reading(_) => invalid(),
        }
        *self = State::Unlocked;
    }
}

fn invalid() -> ! {
    panic!("inconsistent rwlock");
}
