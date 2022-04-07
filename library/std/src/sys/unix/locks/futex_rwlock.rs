use crate::sync::atomic::{
    AtomicI32,
    Ordering::{Acquire, Relaxed, Release},
};
use crate::sys::futex::{futex_wait, futex_wake, futex_wake_all};

pub type MovableRwLock = RwLock;

pub struct RwLock {
    // The state consists of a 30-bit reader counter, a 'readers waiting' flag, and a 'writers waiting' flag.
    // All bits of the reader counter set means write locked.
    // A reader count of zero means the lock is unlocked.
    // See the constants below.
    // Readers wait on this futex.
    state: AtomicI32,
    // The 'condition variable' to notify writers through.
    // Incremented on every signal.
    // Writers wait on this futex.
    writer_notify: AtomicI32,
}

const READ_LOCKED: i32 = 1;
const MAX_READERS: i32 = (1 << 30) - 2;
const WRITE_LOCKED: i32 = (1 << 30) - 1;
const READERS_WAITING: i32 = 1 << 30;
const WRITERS_WAITING: i32 = 1 << 31;

fn readers(state: i32) -> i32 {
    state & !(READERS_WAITING + WRITERS_WAITING)
}

fn readers_waiting(state: i32) -> bool {
    state & READERS_WAITING != 0
}

fn writers_waiting(state: i32) -> bool {
    state & WRITERS_WAITING != 0
}

fn read_lockable(state: i32) -> bool {
    readers(state) < MAX_READERS && !readers_waiting(state) && !writers_waiting(state)
}

impl RwLock {
    #[inline]
    pub const fn new() -> Self {
        Self { state: AtomicI32::new(0), writer_notify: AtomicI32::new(0) }
    }

    #[inline]
    pub unsafe fn destroy(&self) {}

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        self.state
            .fetch_update(Acquire, Relaxed, |s| read_lockable(s).then(|| s + READ_LOCKED))
            .is_ok()
    }

    #[inline]
    pub unsafe fn read(&self) {
        if !self.try_read() {
            self.read_contended();
        }
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        self.state
            .fetch_update(Acquire, Relaxed, |s| (readers(s) == 0).then(|| s + WRITE_LOCKED))
            .is_ok()
    }

    #[inline]
    pub unsafe fn write(&self) {
        if !self.try_write() {
            self.write_contended();
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        let s = self.state.fetch_sub(READ_LOCKED, Release);

        // It's impossible for readers to be waiting if it was read locked.
        debug_assert!(!readers_waiting(s));

        // Wake up a writer if we were the last reader and there's a writer waiting.
        if s == READ_LOCKED + WRITERS_WAITING {
            self.wake_after_read_unlock();
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        if let Err(e) = self.state.compare_exchange(WRITE_LOCKED, 0, Release, Relaxed) {
            // Readers or writers (or both) are waiting.
            self.write_unlock_contended(e);
        }
    }

    #[cold]
    fn read_contended(&self) {
        let mut state = self.spin_read();

        loop {
            // If we can lock it, lock it.
            if read_lockable(state) {
                match self.state.compare_exchange(state, state + READ_LOCKED, Acquire, Relaxed) {
                    Ok(_) => return, // Locked!
                    Err(s) => {
                        state = s;
                        continue;
                    }
                }
            }

            // Check for overflow.
            if readers(state) == MAX_READERS {
                panic!("too many active read locks on RwLock");
            }

            // Make sure the readers waiting bit is set before we go to sleep.
            if !readers_waiting(state) {
                if let Err(s) =
                    self.state.compare_exchange(state, state | READERS_WAITING, Relaxed, Relaxed)
                {
                    state = s;
                    continue;
                }
            }

            // Wait for the state to change.
            futex_wait(&self.state, state | READERS_WAITING, None);

            // Spin again after waking up.
            state = self.spin_read();
        }
    }

    #[cold]
    fn write_contended(&self) {
        let mut state = self.spin_write();

        loop {
            // If it's unlocked, we try to lock it.
            if readers(state) == 0 {
                match self.state.compare_exchange(
                    state,
                    state | WRITE_LOCKED | WRITERS_WAITING, // Other threads might be waiting.
                    Acquire,
                    Relaxed,
                ) {
                    Ok(_) => return, // Locked!
                    Err(s) => {
                        state = s;
                        continue;
                    }
                }
            }

            // Set the waiting bit indicating that we're waiting on it.
            if !writers_waiting(state) {
                if let Err(s) =
                    self.state.compare_exchange(state, state | WRITERS_WAITING, Relaxed, Relaxed)
                {
                    state = s;
                    continue;
                }
            }

            // Examine the notification counter before we check if `state` has changed,
            // to make sure we don't miss any notifications.
            let seq = self.writer_notify.load(Acquire);

            // Don't go to sleep if the lock has become available, or the
            // writers waiting bit is no longer set.
            let s = self.state.load(Relaxed);
            if readers(s) == 0 || !writers_waiting(s) {
                state = s;
                continue;
            }

            // Wait for the state to change.
            futex_wait(&self.writer_notify, seq, None);

            // Spin again after waking up.
            state = self.spin_write();
        }
    }

    #[cold]
    fn wake_after_read_unlock(&self) {
        // If this compare_exchange fails, another writer already locked, which
        // will take care of waking up the next waiting writer.
        if self.state.compare_exchange(WRITERS_WAITING, 0, Relaxed, Relaxed).is_ok() {
            self.writer_notify.fetch_add(1, Release);
            futex_wake(&self.writer_notify);
        }
    }

    #[cold]
    fn write_unlock_contended(&self, mut state: i32) {
        // If there are any waiting writers _or_ waiting readers, but not both (!),
        // we turn off that bit while unlocking.
        if readers_waiting(state) != writers_waiting(state) {
            if self.state.compare_exchange(state, 0, Release, Relaxed).is_err() {
                // The only way this can fail is if the other waiting bit was set too.
                state |= READERS_WAITING | WRITERS_WAITING;
            }
        }

        // If both readers and writers are waiting, unlock but leave the readers waiting.
        if readers_waiting(state) && writers_waiting(state) {
            self.state.store(READERS_WAITING, Release);
        }

        if writers_waiting(state) {
            // Notify one writer, if any writer was waiting.
            self.writer_notify.fetch_add(1, Release);
            if !futex_wake(&self.writer_notify) {
                // If there was no writer to wake up, maybe there's readers to wake up instead.
                if readers_waiting(state) {
                    // If this compare_exchange fails, another writer already locked, which
                    // will take care of waking up the next waiting writer.
                    if self.state.compare_exchange(READERS_WAITING, 0, Relaxed, Relaxed).is_ok() {
                        futex_wake_all(&self.state);
                    }
                }
            }
        } else if readers_waiting(state) {
            // Notify all readers, if any reader was waiting.
            futex_wake_all(&self.state);
        }
    }

    /// Spin for a while, but stop directly at the given condition.
    fn spin_until(&self, f: impl Fn(i32) -> bool) -> i32 {
        let mut spin = 100; // Chosen by fair dice roll.
        loop {
            let state = self.state.load(Relaxed);
            if f(state) || spin == 0 {
                return state;
            }
            crate::hint::spin_loop();
            spin -= 1;
        }
    }

    fn spin_write(&self) -> i32 {
        self.spin_until(|state| {
            // Stop spinning when we can lock it, or when there's waiting
            // writers, to keep things somewhat fair.
            readers(state) == 0 || writers_waiting(state)
        })
    }

    fn spin_read(&self) -> i32 {
        self.spin_until(|state| {
            // Stop spinning when it's unlocked or read locked, or when there's waiting threads.
            readers(state) != WRITE_LOCKED || readers_waiting(state) || writers_waiting(state)
        })
    }
}
