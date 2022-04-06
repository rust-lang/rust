use crate::sync::atomic::{
    AtomicI32,
    Ordering::{Acquire, Relaxed, Release},
};
use crate::sys::futex::{futex_wait, futex_wake, futex_wake_all};

pub type MovableRwLock = RwLock;

pub struct RwLock {
    // The state consists of a 30-bit reader counter, a 'readers waiting' flag, and a 'writers waiting' flag.
    // Bits 0..30:
    //   0: Unlocked
    //   1..=0x3FFF_FFFE: Locked by N readers
    //   0x3FFF_FFFF: Write locked
    // Bit 30: Readers are waiting on this futex.
    // Bit 31: Writers are waiting on the writer_notify futex.
    state: AtomicI32,
    // The 'condition variable' to notify writers through.
    // Incremented on every signal.
    writer_notify: AtomicI32,
}

const READ_LOCKED: i32 = 1;
const MASK: i32 = (1 << 30) - 1;
const WRITE_LOCKED: i32 = MASK;
const MAX_READERS: i32 = MASK - 1;
const READERS_WAITING: i32 = 1 << 30;
const WRITERS_WAITING: i32 = 1 << 31;

fn unlocked(state: i32) -> bool {
    state & MASK == 0
}

fn write_locked(state: i32) -> bool {
    state & MASK == WRITE_LOCKED
}

fn readers_waiting(state: i32) -> bool {
    state & READERS_WAITING != 0
}

fn writers_waiting(state: i32) -> bool {
    state & WRITERS_WAITING != 0
}

fn read_lockable(state: i32) -> bool {
    // This also returns false if the counter could overflow if we tried to read lock it.
    state & MASK < MAX_READERS && !readers_waiting(state) && !writers_waiting(state)
}

fn reached_max_readers(state: i32) -> bool {
    state & MASK == MAX_READERS
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
    pub unsafe fn read_unlock(&self) {
        let state = self.state.fetch_sub(READ_LOCKED, Release) - 1;

        // It's impossible for a reader to be waiting on a read-locked RwLock,
        // except if there is also a writer waiting.
        debug_assert!(!readers_waiting(state) || writers_waiting(state));

        // Wake up a writer if we were the last reader and there's a writer waiting.
        if unlocked(state) && writers_waiting(state) {
            self.wake_writer_or_readers(state);
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
            if reached_max_readers(state) {
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

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        self.state.fetch_update(Acquire, Relaxed, |s| unlocked(s).then(|| s + WRITE_LOCKED)).is_ok()
    }

    #[inline]
    pub unsafe fn write(&self) {
        if !self.try_write() {
            self.write_contended();
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        let state = self.state.fetch_sub(WRITE_LOCKED, Release) - WRITE_LOCKED;

        debug_assert!(unlocked(state));

        if writers_waiting(state) || readers_waiting(state) {
            self.wake_writer_or_readers(state);
        }
    }

    #[cold]
    fn write_contended(&self) {
        let mut state = self.spin_write();

        let mut other_writers_waiting = 0;

        loop {
            // If it's unlocked, we try to lock it.
            if unlocked(state) {
                match self.state.compare_exchange(
                    state,
                    state | WRITE_LOCKED | other_writers_waiting,
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

            // Other writers might be waiting now too, so we should make sure
            // we keep that bit on once we manage lock it.
            other_writers_waiting = WRITERS_WAITING;

            // Examine the notification counter before we check if `state` has changed,
            // to make sure we don't miss any notifications.
            let seq = self.writer_notify.load(Acquire);

            // Don't go to sleep if the lock has become available,
            // or if the writers waiting bit is no longer set.
            let s = self.state.load(Relaxed);
            if unlocked(state) || !writers_waiting(s) {
                state = s;
                continue;
            }

            // Wait for the state to change.
            futex_wait(&self.writer_notify, seq, None);

            // Spin again after waking up.
            state = self.spin_write();
        }
    }

    /// Wake up waiting threads after unlocking.
    ///
    /// If both are waiting, this will wake up only one writer, but will fall
    /// back to waking up readers if there was no writer to wake up.
    #[cold]
    fn wake_writer_or_readers(&self, mut state: i32) {
        assert!(unlocked(state));

        // The readers waiting bit might be turned on at any point now,
        // since readers will block when there's anything waiting.
        // Writers will just lock the lock though, regardless of the waiting bits,
        // so we don't have to worry about the writer waiting bit.
        //
        // If the lock gets locked in the meantime, we don't have to do
        // anything, because then the thread that locked the lock will take
        // care of waking up waiters when it unlocks.

        // If only writers are waiting, wake one of them up.
        if state == WRITERS_WAITING {
            match self.state.compare_exchange(state, 0, Relaxed, Relaxed) {
                Ok(_) => {
                    self.wake_writer();
                    return;
                }
                Err(s) => {
                    // Maybe some readers are now waiting too. So, continue to the next `if`.
                    state = s;
                }
            }
        }

        // If both writers and readers are waiting, leave the readers waiting
        // and only wake up one writer.
        if state == READERS_WAITING + WRITERS_WAITING {
            if self.state.compare_exchange(state, READERS_WAITING, Relaxed, Relaxed).is_err() {
                // The lock got locked. Not our problem anymore.
                return;
            }
            if self.wake_writer() {
                return;
            }
            // No writers were actually waiting. Continue to wake up readers instead.
            state = READERS_WAITING;
        }

        // If readers are waiting, wake them all up.
        if state == READERS_WAITING {
            if self.state.compare_exchange(state, 0, Relaxed, Relaxed).is_ok() {
                futex_wake_all(&self.state);
            }
        }
    }

    fn wake_writer(&self) -> bool {
        self.writer_notify.fetch_add(1, Release);
        futex_wake(&self.writer_notify)
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
        // Stop spinning when it's unlocked or when there's waiting writers, to keep things somewhat fair.
        self.spin_until(|state| unlocked(state) || writers_waiting(state))
    }

    fn spin_read(&self) -> i32 {
        // Stop spinning when it's unlocked or read locked, or when there's waiting threads.
        self.spin_until(|state| {
            !write_locked(state) || readers_waiting(state) || writers_waiting(state)
        })
    }
}
