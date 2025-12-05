use crate::sync::atomic::Ordering::{Acquire, Relaxed, Release};
use crate::sys::futex::{Futex, Primitive, futex_wait, futex_wake, futex_wake_all};

pub struct RwLock {
    // The state consists of a 30-bit reader counter, a 'readers waiting' flag, and a 'writers waiting' flag.
    // Bits 0..30:
    //   0: Unlocked
    //   1..=0x3FFF_FFFE: Locked by N readers
    //   0x3FFF_FFFF: Write locked
    // Bit 30: Readers are waiting on this futex.
    // Bit 31: Writers are waiting on the writer_notify futex.
    state: Futex,
    // The 'condition variable' to notify writers through.
    // Incremented on every signal.
    writer_notify: Futex,
}

const READ_LOCKED: Primitive = 1;
const MASK: Primitive = (1 << 30) - 1;
const WRITE_LOCKED: Primitive = MASK;
const DOWNGRADE: Primitive = READ_LOCKED.wrapping_sub(WRITE_LOCKED); // READ_LOCKED - WRITE_LOCKED
const MAX_READERS: Primitive = MASK - 1;
const READERS_WAITING: Primitive = 1 << 30;
const WRITERS_WAITING: Primitive = 1 << 31;

#[inline]
fn is_unlocked(state: Primitive) -> bool {
    state & MASK == 0
}

#[inline]
fn is_write_locked(state: Primitive) -> bool {
    state & MASK == WRITE_LOCKED
}

#[inline]
fn has_readers_waiting(state: Primitive) -> bool {
    state & READERS_WAITING != 0
}

#[inline]
fn has_writers_waiting(state: Primitive) -> bool {
    state & WRITERS_WAITING != 0
}

#[inline]
fn is_read_lockable(state: Primitive) -> bool {
    // This also returns false if the counter could overflow if we tried to read lock it.
    //
    // We don't allow read-locking if there's readers waiting, even if the lock is unlocked
    // and there's no writers waiting. The only situation when this happens is after unlocking,
    // at which point the unlocking thread might be waking up writers, which have priority over readers.
    // The unlocking thread will clear the readers waiting bit and wake up readers, if necessary.
    state & MASK < MAX_READERS && !has_readers_waiting(state) && !has_writers_waiting(state)
}

#[inline]
fn is_read_lockable_after_wakeup(state: Primitive) -> bool {
    // We make a special case for checking if we can read-lock _after_ a reader thread that went to
    // sleep has been woken up by a call to `downgrade`.
    //
    // `downgrade` will wake up all readers and place the lock in read mode. Thus, there should be
    // no readers waiting and the lock should be read-locked (not write-locked or unlocked).
    //
    // Note that we do not check if any writers are waiting. This is because a call to `downgrade`
    // implies that the caller wants other readers to read the value protected by the lock. If we
    // did not allow readers to acquire the lock before writers after a `downgrade`, then only the
    // original writer would be able to read the value, thus defeating the purpose of `downgrade`.
    state & MASK < MAX_READERS
        && !has_readers_waiting(state)
        && !is_write_locked(state)
        && !is_unlocked(state)
}

#[inline]
fn has_reached_max_readers(state: Primitive) -> bool {
    state & MASK == MAX_READERS
}

impl RwLock {
    #[inline]
    pub const fn new() -> Self {
        Self { state: Futex::new(0), writer_notify: Futex::new(0) }
    }

    #[inline]
    pub fn try_read(&self) -> bool {
        self.state
            .fetch_update(Acquire, Relaxed, |s| is_read_lockable(s).then(|| s + READ_LOCKED))
            .is_ok()
    }

    #[inline]
    pub fn read(&self) {
        let state = self.state.load(Relaxed);
        if !is_read_lockable(state)
            || self
                .state
                .compare_exchange_weak(state, state + READ_LOCKED, Acquire, Relaxed)
                .is_err()
        {
            self.read_contended();
        }
    }

    /// # Safety
    ///
    /// The `RwLock` must be read-locked (N readers) in order to call this.
    #[inline]
    pub unsafe fn read_unlock(&self) {
        let state = self.state.fetch_sub(READ_LOCKED, Release) - READ_LOCKED;

        // It's impossible for a reader to be waiting on a read-locked RwLock,
        // except if there is also a writer waiting.
        debug_assert!(!has_readers_waiting(state) || has_writers_waiting(state));

        // Wake up a writer if we were the last reader and there's a writer waiting.
        if is_unlocked(state) && has_writers_waiting(state) {
            self.wake_writer_or_readers(state);
        }
    }

    #[cold]
    fn read_contended(&self) {
        let mut has_slept = false;
        let mut state = self.spin_read();

        loop {
            // If we have just been woken up, first check for a `downgrade` call.
            // Otherwise, if we can read-lock it, lock it.
            if (has_slept && is_read_lockable_after_wakeup(state)) || is_read_lockable(state) {
                match self.state.compare_exchange_weak(state, state + READ_LOCKED, Acquire, Relaxed)
                {
                    Ok(_) => return, // Locked!
                    Err(s) => {
                        state = s;
                        continue;
                    }
                }
            }

            // Check for overflow.
            assert!(!has_reached_max_readers(state), "too many active read locks on RwLock");

            // Make sure the readers waiting bit is set before we go to sleep.
            if !has_readers_waiting(state) {
                if let Err(s) =
                    self.state.compare_exchange(state, state | READERS_WAITING, Relaxed, Relaxed)
                {
                    state = s;
                    continue;
                }
            }

            // Wait for the state to change.
            futex_wait(&self.state, state | READERS_WAITING, None);
            has_slept = true;

            // Spin again after waking up.
            state = self.spin_read();
        }
    }

    #[inline]
    pub fn try_write(&self) -> bool {
        self.state
            .fetch_update(Acquire, Relaxed, |s| is_unlocked(s).then(|| s + WRITE_LOCKED))
            .is_ok()
    }

    #[inline]
    pub fn write(&self) {
        if self.state.compare_exchange_weak(0, WRITE_LOCKED, Acquire, Relaxed).is_err() {
            self.write_contended();
        }
    }

    /// # Safety
    ///
    /// The `RwLock` must be write-locked (single writer) in order to call this.
    #[inline]
    pub unsafe fn write_unlock(&self) {
        let state = self.state.fetch_sub(WRITE_LOCKED, Release) - WRITE_LOCKED;

        debug_assert!(is_unlocked(state));

        if has_writers_waiting(state) || has_readers_waiting(state) {
            self.wake_writer_or_readers(state);
        }
    }

    /// # Safety
    ///
    /// The `RwLock` must be write-locked (single writer) in order to call this.
    #[inline]
    pub unsafe fn downgrade(&self) {
        // Removes all write bits and adds a single read bit.
        let state = self.state.fetch_add(DOWNGRADE, Release);
        debug_assert!(is_write_locked(state), "RwLock must be write locked to call `downgrade`");

        if has_readers_waiting(state) {
            // Since we had the exclusive lock, nobody else can unset this bit.
            self.state.fetch_sub(READERS_WAITING, Relaxed);
            futex_wake_all(&self.state);
        }
    }

    #[cold]
    fn write_contended(&self) {
        let mut state = self.spin_write();

        let mut other_writers_waiting = 0;

        loop {
            // If it's unlocked, we try to lock it.
            if is_unlocked(state) {
                match self.state.compare_exchange_weak(
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
            if !has_writers_waiting(state) {
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
            state = self.state.load(Relaxed);
            if is_unlocked(state) || !has_writers_waiting(state) {
                continue;
            }

            // Wait for the state to change.
            futex_wait(&self.writer_notify, seq, None);

            // Spin again after waking up.
            state = self.spin_write();
        }
    }

    /// Wakes up waiting threads after unlocking.
    ///
    /// If both are waiting, this will wake up only one writer, but will fall
    /// back to waking up readers if there was no writer to wake up.
    #[cold]
    fn wake_writer_or_readers(&self, mut state: Primitive) {
        assert!(is_unlocked(state));

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
            // No writers were actually blocked on futex_wait, so we continue
            // to wake up readers instead, since we can't be sure if we notified a writer.
            state = READERS_WAITING;
        }

        // If readers are waiting, wake them all up.
        if state == READERS_WAITING {
            if self.state.compare_exchange(state, 0, Relaxed, Relaxed).is_ok() {
                futex_wake_all(&self.state);
            }
        }
    }

    /// This wakes one writer and returns true if we woke up a writer that was
    /// blocked on futex_wait.
    ///
    /// If this returns false, it might still be the case that we notified a
    /// writer that was about to go to sleep.
    fn wake_writer(&self) -> bool {
        self.writer_notify.fetch_add(1, Release);
        futex_wake(&self.writer_notify)
        // Note that FreeBSD and DragonFlyBSD don't tell us whether they woke
        // up any threads or not, and always return `false` here. That still
        // results in correct behavior: it just means readers get woken up as
        // well in case both readers and writers were waiting.
    }

    /// Spin for a while, but stop directly at the given condition.
    #[inline]
    fn spin_until(&self, f: impl Fn(Primitive) -> bool) -> Primitive {
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

    #[inline]
    fn spin_write(&self) -> Primitive {
        // Stop spinning when it's unlocked or when there's waiting writers, to keep things somewhat fair.
        self.spin_until(|state| is_unlocked(state) || has_writers_waiting(state))
    }

    #[inline]
    fn spin_read(&self) -> Primitive {
        // Stop spinning when it's unlocked or read locked, or when there's waiting threads.
        self.spin_until(|state| {
            !is_write_locked(state) || has_readers_waiting(state) || has_writers_waiting(state)
        })
    }
}
