//! Nintendo 3DS-specific extensions to primitives in the [`std::thread`] module.
//!
//! All 3DS models have at least two CPU cores available to spawn threads on:
//! The application core (appcore) and the system core (syscore). The New 3DS
//! has an additional two cores, the first of which can also run user-created
//! threads.
//!
//! Threads spawned on the appcore are cooperative rather than preemptive. This
//! means that threads must explicitly yield control to other threads (whether
//! via synchronization primitives or explicit calls to `yield_now`) when they
//! are not actively performing work. Failure to do so may result in control
//! flow being stuck in an inactive thread while the other threads are powerless
//! to continue their work.
//!
//! However, it is possible to spawn one fully preemptive thread on the syscore
//! by using a service call to reserve a slice of time for a thread to run.
//! Attempting to run more than one thread at a time on the syscore will result
//! in an error.
//!
//! [`std::thread`]: crate::thread

#![unstable(feature = "horizon_thread_ext", issue = "none")]

/// Extensions on [`std::thread::Builder`] for the Nintendo 3DS.
///
/// [`std::thread::Builder`]: crate::thread::Builder
pub trait BuilderExt: Sized {
    /// Sets the priority level for the new thread.
    ///
    /// Low values gives the thread higher priority. For userland apps, this has
    /// to be within the range of 0x18 to 0x3F inclusive. The main thread
    /// usually has a priority of 0x30, but not always.
    fn priority(self, priority: i32) -> Self;

    /// Sets the ID of the processor the thread should be run on. Threads on the 3DS are only
    /// preemptive if they are on the system core. Otherwise they are cooperative (must yield to let
    /// other threads run).
    ///
    /// Processor IDs are labeled starting from 0. On Old3DS it must be <2, and
    /// on New3DS it must be <4. Pass -1 to execute the thread on all CPUs and
    /// -2 to execute the thread on the default CPU (set in the application's Exheader).
    ///
    /// * Processor #0 is the application core. It is always possible to create
    ///   a thread on this core.
    /// * Processor #1 is the system core. If the CPU time limit is set, it is
    ///   possible to create a single thread on this core.
    /// * Processor #2 is New3DS exclusive. Normal applications can create
    ///   threads on this core only if the built application has proper external setup.
    /// * Processor #3 is New3DS exclusive. Normal applications cannot create
    ///   threads on this core.
    fn processor_id(self, processor_id: i32) -> Self;
}

impl BuilderExt for crate::thread::Builder {
    fn priority(mut self, priority: i32) -> Self {
        self.native_options.priority = Some(priority);
        self
    }

    fn processor_id(mut self, processor_id: i32) -> Self {
        self.native_options.processor_id = Some(processor_id);
        self
    }
}

/// Get the current thread's priority level. Lower values correspond to higher
/// priority levels.
pub fn current_priority() -> i32 {
    let thread_id = unsafe { libc::pthread_self() };
    let mut policy = 0;
    let mut sched_param = libc::sched_param { sched_priority: 0 };

    let result = unsafe { libc::pthread_getschedparam(thread_id, &mut policy, &mut sched_param) };
    assert_eq!(result, 0);

    sched_param.sched_priority
}
