//! Code that decides when workers should go to sleep. See README.md
//! for an overview.

use std::sync::atomic::Ordering;
use std::sync::{Condvar, Mutex};
use std::thread;

use crossbeam_utils::CachePadded;

use crate::DeadlockHandler;
use crate::latch::CoreLatch;
use crate::registry::WorkerThread;

mod counters;
pub(crate) use self::counters::THREADS_MAX;
use self::counters::{AtomicCounters, JobsEventCounter};

struct SleepData {
    /// The number of threads in the thread pool.
    worker_count: usize,

    /// The number of threads in the thread pool which are running and
    /// aren't blocked in user code or sleeping.
    active_threads: usize,

    /// The number of threads which are blocked in user code.
    /// This doesn't include threads blocked by this module.
    blocked_threads: usize,
}

impl SleepData {
    /// Checks if the conditions for a deadlock holds and if so calls the deadlock handler
    #[inline]
    pub(super) fn deadlock_check(&self, deadlock_handler: &Option<Box<DeadlockHandler>>) {
        if self.active_threads == 0 && self.blocked_threads > 0 {
            (deadlock_handler.as_ref().unwrap())();
        }
    }
}

/// The `Sleep` struct is embedded into each registry. It governs the waking and sleeping
/// of workers. It has callbacks that are invoked periodically at significant events,
/// such as when workers are looping and looking for work, when latches are set, or when
/// jobs are published, and it either blocks threads or wakes them in response to these
/// events. See the [`README.md`] in this module for more details.
///
/// [`README.md`] README.md
pub(super) struct Sleep {
    /// One "sleep state" per worker. Used to track if a worker is sleeping and to have
    /// them block.
    worker_sleep_states: Vec<CachePadded<WorkerSleepState>>,

    counters: AtomicCounters,

    data: Mutex<SleepData>,
}

/// An instance of this struct is created when a thread becomes idle.
/// It is consumed when the thread finds work, and passed by `&mut`
/// reference for operations that preserve the idle state. (In other
/// words, producing one of these structs is evidence the thread is
/// idle.) It tracks state such as how long the thread has been idle.
pub(super) struct IdleState {
    /// What is worker index of the idle thread?
    worker_index: usize,

    /// How many rounds have we been circling without sleeping?
    rounds: u32,

    /// Once we become sleepy, what was the sleepy counter value?
    /// Set to `INVALID_SLEEPY_COUNTER` otherwise.
    jobs_counter: JobsEventCounter,
}

/// The "sleep state" for an individual worker.
#[derive(Default)]
struct WorkerSleepState {
    /// Set to true when the worker goes to sleep; set to false when
    /// the worker is notified or when it wakes.
    is_blocked: Mutex<bool>,

    condvar: Condvar,
}

const ROUNDS_UNTIL_SLEEPY: u32 = 32;
const ROUNDS_UNTIL_SLEEPING: u32 = ROUNDS_UNTIL_SLEEPY + 1;

impl Sleep {
    pub(super) fn new(n_threads: usize) -> Sleep {
        assert!(n_threads <= THREADS_MAX);
        Sleep {
            worker_sleep_states: (0..n_threads).map(|_| Default::default()).collect(),
            counters: AtomicCounters::new(),
            data: Mutex::new(SleepData {
                worker_count: n_threads,
                active_threads: n_threads,
                blocked_threads: 0,
            }),
        }
    }

    /// Mark a Rayon worker thread as blocked. This triggers the deadlock handler
    /// if no other worker thread is active
    #[inline]
    pub(super) fn mark_blocked(&self, deadlock_handler: &Option<Box<DeadlockHandler>>) {
        let mut data = self.data.lock().unwrap();
        debug_assert!(data.active_threads > 0);
        debug_assert!(data.blocked_threads < data.worker_count);
        debug_assert!(data.active_threads > 0);
        data.active_threads -= 1;
        data.blocked_threads += 1;

        data.deadlock_check(deadlock_handler);
    }

    /// Mark a previously blocked Rayon worker thread as unblocked
    #[inline]
    pub(super) fn mark_unblocked(&self) {
        let mut data = self.data.lock().unwrap();
        debug_assert!(data.active_threads < data.worker_count);
        debug_assert!(data.blocked_threads > 0);
        data.active_threads += 1;
        data.blocked_threads -= 1;
    }

    #[inline]
    pub(super) fn start_looking(&self, worker_index: usize) -> IdleState {
        self.counters.add_inactive_thread();

        IdleState { worker_index, rounds: 0, jobs_counter: JobsEventCounter::DUMMY }
    }

    #[inline]
    pub(super) fn work_found(&self) {
        // If we were the last idle thread and other threads are still sleeping,
        // then we should wake up another thread.
        let threads_to_wake = self.counters.sub_inactive_thread();
        self.wake_any_threads(threads_to_wake as u32);
    }

    #[inline]
    pub(super) fn no_work_found(
        &self,
        idle_state: &mut IdleState,
        latch: &CoreLatch,
        thread: &WorkerThread,
    ) {
        if idle_state.rounds < ROUNDS_UNTIL_SLEEPY {
            thread::yield_now();
            idle_state.rounds += 1;
        } else if idle_state.rounds == ROUNDS_UNTIL_SLEEPY {
            idle_state.jobs_counter = self.announce_sleepy();
            idle_state.rounds += 1;
            thread::yield_now();
        } else if idle_state.rounds < ROUNDS_UNTIL_SLEEPING {
            idle_state.rounds += 1;
            thread::yield_now();
        } else {
            debug_assert_eq!(idle_state.rounds, ROUNDS_UNTIL_SLEEPING);
            self.sleep(idle_state, latch, thread);
        }
    }

    #[cold]
    fn announce_sleepy(&self) -> JobsEventCounter {
        self.counters.increment_jobs_event_counter_if(JobsEventCounter::is_active).jobs_counter()
    }

    #[cold]
    fn sleep(&self, idle_state: &mut IdleState, latch: &CoreLatch, thread: &WorkerThread) {
        let worker_index = idle_state.worker_index;

        if !latch.get_sleepy() {
            return;
        }

        let sleep_state = &self.worker_sleep_states[worker_index];
        let mut is_blocked = sleep_state.is_blocked.lock().unwrap();
        debug_assert!(!*is_blocked);

        // Our latch was signalled. We should wake back up fully as we
        // will have some stuff to do.
        if !latch.fall_asleep() {
            idle_state.wake_fully();
            return;
        }

        loop {
            let counters = self.counters.load(Ordering::SeqCst);

            // Check if the JEC has changed since we got sleepy.
            debug_assert!(idle_state.jobs_counter.is_sleepy());
            if counters.jobs_counter() != idle_state.jobs_counter {
                // JEC has changed, so a new job was posted, but for some reason
                // we didn't see it. We should return to just before the SLEEPY
                // state so we can do another search and (if we fail to find
                // work) go back to sleep.
                idle_state.wake_partly();
                latch.wake_up();
                return;
            }

            // Otherwise, let's move from IDLE to SLEEPING.
            if self.counters.try_add_sleeping_thread(counters) {
                break;
            }
        }

        // Successfully registered as asleep.

        // We have one last check for injected jobs to do. This protects against
        // deadlock in the very unlikely event that
        //
        // - an external job is being injected while we are sleepy
        // - that job triggers the rollover over the JEC such that we don't see it
        // - we are the last active worker thread
        std::sync::atomic::fence(Ordering::SeqCst);
        if thread.has_injected_job() {
            // If we see an externally injected job, then we have to 'wake
            // ourselves up'. (Ordinarily, `sub_sleeping_thread` is invoked by
            // the one that wakes us.)
            self.counters.sub_sleeping_thread();
        } else {
            {
                // Decrement the number of active threads and check for a deadlock
                let mut data = self.data.lock().unwrap();
                data.active_threads -= 1;
                data.deadlock_check(&thread.registry.deadlock_handler);
            }

            // If we don't see an injected job (the normal case), then flag
            // ourselves as asleep and wait till we are notified.
            //
            // (Note that `is_blocked` is held under a mutex and the mutex was
            // acquired *before* we incremented the "sleepy counter". This means
            // that whomever is coming to wake us will have to wait until we
            // release the mutex in the call to `wait`, so they will see this
            // boolean as true.)
            thread.registry.release_thread();
            *is_blocked = true;
            while *is_blocked {
                is_blocked = sleep_state.condvar.wait(is_blocked).unwrap();
            }

            // Drop `is_blocked` now in case `acquire_thread` blocks
            drop(is_blocked);

            thread.registry.acquire_thread();
        }

        // Update other state:
        idle_state.wake_fully();
        latch.wake_up();
    }

    /// Notify the given thread that it should wake up (if it is
    /// sleeping). When this method is invoked, we typically know the
    /// thread is asleep, though in rare cases it could have been
    /// awoken by (e.g.) new work having been posted.
    pub(super) fn notify_worker_latch_is_set(&self, target_worker_index: usize) {
        self.wake_specific_thread(target_worker_index);
    }

    /// Signals that `num_jobs` new jobs were injected into the thread
    /// pool from outside. This function will ensure that there are
    /// threads available to process them, waking threads from sleep
    /// if necessary.
    ///
    /// # Parameters
    ///
    /// - `num_jobs` -- lower bound on number of jobs available for stealing.
    ///   We'll try to get at least one thread per job.
    #[inline]
    pub(super) fn new_injected_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        // This fence is needed to guarantee that threads
        // as they are about to fall asleep, observe any
        // new jobs that may have been injected.
        std::sync::atomic::fence(Ordering::SeqCst);

        self.new_jobs(num_jobs, queue_was_empty)
    }

    /// Signals that `num_jobs` new jobs were pushed onto a thread's
    /// local deque. This function will try to ensure that there are
    /// threads available to process them, waking threads from sleep
    /// if necessary. However, this is not guaranteed: under certain
    /// race conditions, the function may fail to wake any new
    /// threads; in that case the existing thread should eventually
    /// pop the job.
    ///
    /// # Parameters
    ///
    /// - `num_jobs` -- lower bound on number of jobs available for stealing.
    ///   We'll try to get at least one thread per job.
    #[inline]
    pub(super) fn new_internal_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        self.new_jobs(num_jobs, queue_was_empty)
    }

    /// Common helper for `new_injected_jobs` and `new_internal_jobs`.
    #[inline]
    fn new_jobs(&self, num_jobs: u32, queue_was_empty: bool) {
        // Read the counters and -- if sleepy workers have announced themselves
        // -- announce that there is now work available. The final value of `counters`
        // with which we exit the loop thus corresponds to a state when
        let counters = self.counters.increment_jobs_event_counter_if(JobsEventCounter::is_sleepy);
        let num_awake_but_idle = counters.awake_but_idle_threads();
        let num_sleepers = counters.sleeping_threads();

        if num_sleepers == 0 {
            // nobody to wake
            return;
        }

        // Promote from u16 to u32 so we can interoperate with
        // num_jobs more easily.
        let num_awake_but_idle = num_awake_but_idle as u32;
        let num_sleepers = num_sleepers as u32;

        // If the queue is non-empty, then we always wake up a worker
        // -- clearly the existing idle jobs aren't enough. Otherwise,
        // check to see if we have enough idle workers.
        if !queue_was_empty {
            let num_to_wake = Ord::min(num_jobs, num_sleepers);
            self.wake_any_threads(num_to_wake);
        } else if num_awake_but_idle < num_jobs {
            let num_to_wake = Ord::min(num_jobs - num_awake_but_idle, num_sleepers);
            self.wake_any_threads(num_to_wake);
        }
    }

    #[cold]
    fn wake_any_threads(&self, mut num_to_wake: u32) {
        if num_to_wake > 0 {
            for i in 0..self.worker_sleep_states.len() {
                if self.wake_specific_thread(i) {
                    num_to_wake -= 1;
                    if num_to_wake == 0 {
                        return;
                    }
                }
            }
        }
    }

    fn wake_specific_thread(&self, index: usize) -> bool {
        let sleep_state = &self.worker_sleep_states[index];

        let mut is_blocked = sleep_state.is_blocked.lock().unwrap();
        if *is_blocked {
            *is_blocked = false;

            // Increment the number of active threads
            self.data.lock().unwrap().active_threads += 1;

            sleep_state.condvar.notify_one();

            // When the thread went to sleep, it will have incremented
            // this value. When we wake it, its our job to decrement
            // it. We could have the thread do it, but that would
            // introduce a delay between when the thread was
            // *notified* and when this counter was decremented. That
            // might mislead people with new work into thinking that
            // there are sleeping threads that they should try to
            // wake, when in fact there is nothing left for them to
            // do.
            self.counters.sub_sleeping_thread();

            true
        } else {
            false
        }
    }
}

impl IdleState {
    fn wake_fully(&mut self) {
        self.rounds = 0;
        self.jobs_counter = JobsEventCounter::DUMMY;
    }

    fn wake_partly(&mut self) {
        self.rounds = ROUNDS_UNTIL_SLEEPY;
        self.jobs_counter = JobsEventCounter::DUMMY;
    }
}
