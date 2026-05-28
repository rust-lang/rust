use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct AtomicCounters {
    /// Packs together a number of counters. The counters are ordered as
    /// follows, from least to most significant bits (here, we assuming
    /// that [`THREADS_BITS`] is equal to 10):
    ///
    /// * Bits 0..10: Stores the number of **sleeping threads**
    /// * Bits 10..20: Stores the number of **inactive threads**
    /// * Bits 20..: Stores the **job event counter** (JEC)
    ///
    /// This uses 10 bits ([`THREADS_BITS`]) to encode the number of threads. Note
    /// that the total number of bits (and hence the number of bits used for the
    /// JEC) will depend on whether we are using a 32- or 64-bit architecture.
    value: AtomicUsize,
}

#[derive(Copy, Clone)]
pub(super) struct Counters {
    word: usize,
}

/// A value read from the **Jobs Event Counter**.
/// See the [`README.md`](README.md) for more
/// coverage of how the jobs event counter works.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub(super) struct JobsEventCounter(usize);

impl JobsEventCounter {
    pub(super) const DUMMY: JobsEventCounter = JobsEventCounter(usize::MAX);

    #[inline]
    pub(super) fn as_usize(self) -> usize {
        self.0
    }

    /// The JEC "is sleepy" if the last thread to increment it was in the
    /// process of becoming sleepy. This is indicated by its value being *even*.
    /// When new jobs are posted, they check if the JEC is sleepy, and if so
    /// they incremented it.
    #[inline]
    pub(super) fn is_sleepy(self) -> bool {
        (self.as_usize() & 1) == 0
    }

    /// The JEC "is active" if the last thread to increment it was posting new
    /// work. This is indicated by its value being *odd*. When threads get
    /// sleepy, they will check if the JEC is active, and increment it.
    #[inline]
    pub(super) fn is_active(self) -> bool {
        !self.is_sleepy()
    }
}

/// Number of bits used for the thread counters.
#[cfg(target_pointer_width = "64")]
const THREADS_BITS: usize = 16;

#[cfg(target_pointer_width = "32")]
const THREADS_BITS: usize = 8;

/// Bits to shift to select the sleeping threads
/// (used with `select_bits`).
#[allow(clippy::erasing_op)]
const SLEEPING_SHIFT: usize = 0 * THREADS_BITS;

/// Bits to shift to select the inactive threads
/// (used with `select_bits`).
#[allow(clippy::identity_op)]
const INACTIVE_SHIFT: usize = 1 * THREADS_BITS;

/// Bits to shift to select the JEC
/// (use JOBS_BITS).
const JEC_SHIFT: usize = 2 * THREADS_BITS;

/// Max value for the thread counters.
pub(crate) const THREADS_MAX: usize = (1 << THREADS_BITS) - 1;

/// Constant that can be added to add one sleeping thread.
const ONE_SLEEPING: usize = 1;

/// Constant that can be added to add one inactive thread.
/// An inactive thread is either idle, sleepy, or sleeping.
const ONE_INACTIVE: usize = 1 << INACTIVE_SHIFT;

/// Constant that can be added to add one to the JEC.
const ONE_JEC: usize = 1 << JEC_SHIFT;

impl AtomicCounters {
    #[inline]
    pub(super) fn new() -> AtomicCounters {
        AtomicCounters { value: AtomicUsize::new(0) }
    }

    /// Load and return the current value of the various counters.
    /// This value can then be given to other method which will
    /// attempt to update the counters via compare-and-swap.
    #[inline]
    pub(super) fn load(&self, ordering: Ordering) -> Counters {
        Counters::new(self.value.load(ordering))
    }

    #[inline]
    fn try_exchange(&self, old_value: Counters, new_value: Counters, ordering: Ordering) -> bool {
        self.value
            .compare_exchange(old_value.word, new_value.word, ordering, Ordering::Relaxed)
            .is_ok()
    }

    /// Adds an inactive thread. This cannot fail.
    ///
    /// This should be invoked when a thread enters its idle loop looking
    /// for work. It is decremented when work is found. Note that it is
    /// not decremented if the thread transitions from idle to sleepy or sleeping;
    /// so the number of inactive threads is always greater-than-or-equal
    /// to the number of sleeping threads.
    #[inline]
    pub(super) fn add_inactive_thread(&self) {
        self.value.fetch_add(ONE_INACTIVE, Ordering::SeqCst);
    }

    /// Increments the jobs event counter if `increment_when`, when applied to
    /// the current value, is true. Used to toggle the JEC from even (sleepy) to
    /// odd (active) or vice versa. Returns the final value of the counters, for
    /// which `increment_when` is guaranteed to return false.
    pub(super) fn increment_jobs_event_counter_if(
        &self,
        increment_when: impl Fn(JobsEventCounter) -> bool,
    ) -> Counters {
        loop {
            let old_value = self.load(Ordering::SeqCst);
            if increment_when(old_value.jobs_counter()) {
                let new_value = old_value.increment_jobs_counter();
                if self.try_exchange(old_value, new_value, Ordering::SeqCst) {
                    return new_value;
                }
            } else {
                return old_value;
            }
        }
    }

    /// Subtracts an inactive thread. This cannot fail. It is invoked
    /// when a thread finds work and hence becomes active. It returns the
    /// number of sleeping threads to wake up (if any).
    ///
    /// See `add_inactive_thread`.
    #[inline]
    pub(super) fn sub_inactive_thread(&self) -> usize {
        let old_value = Counters::new(self.value.fetch_sub(ONE_INACTIVE, Ordering::SeqCst));
        debug_assert!(
            old_value.inactive_threads() > 0,
            "sub_inactive_thread: old_value {:?} has no inactive threads",
            old_value,
        );
        debug_assert!(
            old_value.sleeping_threads() <= old_value.inactive_threads(),
            "sub_inactive_thread: old_value {:?} had {} sleeping threads and {} inactive threads",
            old_value,
            old_value.sleeping_threads(),
            old_value.inactive_threads(),
        );

        // Current heuristic: whenever an inactive thread goes away, if
        // there are any sleeping threads, wake 'em up.
        let sleeping_threads = old_value.sleeping_threads();
        Ord::min(sleeping_threads, 2)
    }

    /// Subtracts a sleeping thread. This cannot fail, but it is only
    /// safe to do if you you know the number of sleeping threads is
    /// non-zero (i.e., because you have just awoken a sleeping
    /// thread).
    #[inline]
    pub(super) fn sub_sleeping_thread(&self) {
        let old_value = Counters::new(self.value.fetch_sub(ONE_SLEEPING, Ordering::SeqCst));
        debug_assert!(
            old_value.sleeping_threads() > 0,
            "sub_sleeping_thread: old_value {:?} had no sleeping threads",
            old_value,
        );
        debug_assert!(
            old_value.sleeping_threads() <= old_value.inactive_threads(),
            "sub_sleeping_thread: old_value {:?} had {} sleeping threads and {} inactive threads",
            old_value,
            old_value.sleeping_threads(),
            old_value.inactive_threads(),
        );
    }

    #[inline]
    pub(super) fn try_add_sleeping_thread(&self, old_value: Counters) -> bool {
        debug_assert!(
            old_value.inactive_threads() > 0,
            "try_add_sleeping_thread: old_value {:?} has no inactive threads",
            old_value,
        );
        debug_assert!(
            old_value.sleeping_threads() < THREADS_MAX,
            "try_add_sleeping_thread: old_value {:?} has too many sleeping threads",
            old_value,
        );

        let mut new_value = old_value;
        new_value.word += ONE_SLEEPING;

        self.try_exchange(old_value, new_value, Ordering::SeqCst)
    }
}

#[inline]
fn select_thread(word: usize, shift: usize) -> usize {
    (word >> shift) & THREADS_MAX
}

#[inline]
fn select_jec(word: usize) -> usize {
    word >> JEC_SHIFT
}

impl Counters {
    #[inline]
    fn new(word: usize) -> Counters {
        Counters { word }
    }

    #[inline]
    fn increment_jobs_counter(self) -> Counters {
        // We can freely add to JEC because it occupies the most significant bits.
        // Thus it doesn't overflow into the other counters, just wraps itself.
        Counters { word: self.word.wrapping_add(ONE_JEC) }
    }

    #[inline]
    pub(super) fn jobs_counter(self) -> JobsEventCounter {
        JobsEventCounter(select_jec(self.word))
    }

    /// The number of threads that are not actively
    /// executing work. They may be idle, sleepy, or asleep.
    #[inline]
    pub(super) fn inactive_threads(self) -> usize {
        select_thread(self.word, INACTIVE_SHIFT)
    }

    #[inline]
    pub(super) fn awake_but_idle_threads(self) -> usize {
        debug_assert!(
            self.sleeping_threads() <= self.inactive_threads(),
            "sleeping threads: {} > raw idle threads {}",
            self.sleeping_threads(),
            self.inactive_threads()
        );
        self.inactive_threads() - self.sleeping_threads()
    }

    #[inline]
    pub(super) fn sleeping_threads(self) -> usize {
        select_thread(self.word, SLEEPING_SHIFT)
    }
}

impl std::fmt::Debug for Counters {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let word = format!("{:016x}", self.word);
        fmt.debug_struct("Counters")
            .field("word", &word)
            .field("jobs", &self.jobs_counter().0)
            .field("inactive", &self.inactive_threads())
            .field("sleeping", &self.sleeping_threads())
            .finish()
    }
}
