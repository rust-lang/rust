// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Abstraction of a thread pool for basic parallelism.

#![unstable(feature = "std_misc",
            reason = "the semantics of a failing task and whether a thread is \
                      re-attached to a thread pool are somewhat unclear, and the \
                      utility of this type in `std::sync` is questionable with \
                      respect to the jobs of other primitives")]

use core::prelude::*;

use sync::{Arc, Mutex};
use sync::mpsc::{channel, Sender, Receiver};
use thread;
use thunk::Thunk;

struct Sentinel<'a> {
    jobs: &'a Arc<Mutex<Receiver<Thunk<'static>>>>,
    active: bool
}

impl<'a> Sentinel<'a> {
    fn new(jobs: &'a Arc<Mutex<Receiver<Thunk<'static>>>>) -> Sentinel<'a> {
        Sentinel {
            jobs: jobs,
            active: true
        }
    }

    // Cancel and destroy this sentinel.
    fn cancel(mut self) {
        self.active = false;
    }
}

#[unsafe_destructor]
impl<'a> Drop for Sentinel<'a> {
    fn drop(&mut self) {
        if self.active {
            spawn_in_pool(self.jobs.clone())
        }
    }
}

/// A thread pool used to execute functions in parallel.
///
/// Spawns `n` worker threads and replenishes the pool if any worker threads
/// panic.
///
/// # Example
///
/// ```rust
/// use std::sync::TaskPool;
/// use std::iter::AdditiveIterator;
/// use std::sync::mpsc::channel;
///
/// let pool = TaskPool::new(4);
///
/// let (tx, rx) = channel();
/// for _ in 0..8 {
///     let tx = tx.clone();
///     pool.execute(move|| {
///         tx.send(1_u32).unwrap();
///     });
/// }
///
/// assert_eq!(rx.iter().take(8).sum(), 8);
/// ```
pub struct TaskPool {
    // How the threadpool communicates with subthreads.
    //
    // This is the only such Sender, so when it is dropped all subthreads will
    // quit.
    jobs: Sender<Thunk<'static>>
}

impl TaskPool {
    /// Spawns a new thread pool with `threads` threads.
    ///
    /// # Panics
    ///
    /// This function will panic if `threads` is 0.
    pub fn new(threads: uint) -> TaskPool {
        assert!(threads >= 1);

        let (tx, rx) = channel::<Thunk>();
        let rx = Arc::new(Mutex::new(rx));

        // Threadpool threads
        for _ in 0..threads {
            spawn_in_pool(rx.clone());
        }

        TaskPool { jobs: tx }
    }

    /// Executes the function `job` on a thread in the pool.
    pub fn execute<F>(&self, job: F)
        where F : FnOnce(), F : Send + 'static
    {
        self.jobs.send(Thunk::new(job)).unwrap();
    }
}

fn spawn_in_pool(jobs: Arc<Mutex<Receiver<Thunk<'static>>>>) {
    thread::spawn(move || {
        // Will spawn a new thread on panic unless it is cancelled.
        let sentinel = Sentinel::new(&jobs);

        loop {
            let message = {
                // Only lock jobs for the time it takes
                // to get a job, not run it.
                let lock = jobs.lock().unwrap();
                lock.recv()
            };

            match message {
                Ok(job) => job.invoke(()),

                // The Taskpool was dropped.
                Err(..) => break
            }
        }

        sentinel.cancel();
    });
}

#[cfg(test)]
mod test {
    use prelude::v1::*;
    use super::*;
    use sync::mpsc::channel;

    const TEST_TASKS: uint = 4;

    #[test]
    fn test_works() {
        use iter::AdditiveIterator;

        let pool = TaskPool::new(TEST_TASKS);

        let (tx, rx) = channel();
        for _ in 0..TEST_TASKS {
            let tx = tx.clone();
            pool.execute(move|| {
                tx.send(1).unwrap();
            });
        }

        assert_eq!(rx.iter().take(TEST_TASKS).sum(), TEST_TASKS);
    }

    #[test]
    #[should_fail]
    fn test_zero_tasks_panic() {
        TaskPool::new(0);
    }

    #[test]
    fn test_recovery_from_subtask_panic() {
        use iter::AdditiveIterator;

        let pool = TaskPool::new(TEST_TASKS);

        // Panic all the existing threads.
        for _ in 0..TEST_TASKS {
            pool.execute(move|| -> () { panic!() });
        }

        // Ensure new threads were spawned to compensate.
        let (tx, rx) = channel();
        for _ in 0..TEST_TASKS {
            let tx = tx.clone();
            pool.execute(move|| {
                tx.send(1).unwrap();
            });
        }

        assert_eq!(rx.iter().take(TEST_TASKS).sum(), TEST_TASKS);
    }

    #[test]
    fn test_should_not_panic_on_drop_if_subtasks_panic_after_drop() {
        use sync::{Arc, Barrier};

        let pool = TaskPool::new(TEST_TASKS);
        let waiter = Arc::new(Barrier::new(TEST_TASKS + 1));

        // Panic all the existing threads in a bit.
        for _ in 0..TEST_TASKS {
            let waiter = waiter.clone();
            pool.execute(move|| {
                waiter.wait();
                panic!();
            });
        }

        drop(pool);

        // Kick off the failure.
        waiter.wait();
    }
}
