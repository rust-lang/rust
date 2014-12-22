// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Abstraction of a task pool for basic parallelism.

use core::prelude::*;

use thread::Thread;
use comm::{channel, Sender, Receiver};
use sync::{Arc, Mutex};
use thunk::Thunk;

struct Sentinel<'a> {
    jobs: &'a Arc<Mutex<Receiver<Thunk>>>,
    active: bool
}

impl<'a> Sentinel<'a> {
    fn new(jobs: &Arc<Mutex<Receiver<Thunk>>>) -> Sentinel {
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

/// A task pool used to execute functions in parallel.
///
/// Spawns `n` worker tasks and replenishes the pool if any worker tasks
/// panic.
///
/// # Example
///
/// ```rust
/// # use std::sync::TaskPool;
/// # use std::iter::AdditiveIterator;
///
/// let pool = TaskPool::new(4u);
///
/// let (tx, rx) = channel();
/// for _ in range(0, 8u) {
///     let tx = tx.clone();
///     pool.execute(move|| {
///         tx.send(1u);
///     });
/// }
///
/// assert_eq!(rx.iter().take(8u).sum(), 8u);
/// ```
pub struct TaskPool {
    // How the taskpool communicates with subtasks.
    //
    // This is the only such Sender, so when it is dropped all subtasks will
    // quit.
    jobs: Sender<Thunk>
}

impl TaskPool {
    /// Spawns a new task pool with `tasks` tasks.
    ///
    /// # Panics
    ///
    /// This function will panic if `tasks` is 0.
    pub fn new(tasks: uint) -> TaskPool {
        assert!(tasks >= 1);

        let (tx, rx) = channel::<Thunk>();
        let rx = Arc::new(Mutex::new(rx));

        // Taskpool tasks.
        for _ in range(0, tasks) {
            spawn_in_pool(rx.clone());
        }

        TaskPool { jobs: tx }
    }

    /// Executes the function `job` on a task in the pool.
    pub fn execute<F>(&self, job: F)
        where F : FnOnce(), F : Send
    {
        self.jobs.send(Thunk::new(job));
    }
}

fn spawn_in_pool(jobs: Arc<Mutex<Receiver<Thunk>>>) {
    Thread::spawn(move |:| {
        // Will spawn a new task on panic unless it is cancelled.
        let sentinel = Sentinel::new(&jobs);

        loop {
            let message = {
                // Only lock jobs for the time it takes
                // to get a job, not run it.
                let lock = jobs.lock();
                lock.recv_opt()
            };

            match message {
                Ok(job) => job.invoke(()),

                // The Taskpool was dropped.
                Err(..) => break
            }
        }

        sentinel.cancel();
    }).detach();
}

#[cfg(test)]
mod test {
    use prelude::*;
    use super::*;

    const TEST_TASKS: uint = 4u;

    #[test]
    fn test_works() {
        use iter::AdditiveIterator;

        let pool = TaskPool::new(TEST_TASKS);

        let (tx, rx) = channel();
        for _ in range(0, TEST_TASKS) {
            let tx = tx.clone();
            pool.execute(move|| {
                tx.send(1u);
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

        // Panic all the existing tasks.
        for _ in range(0, TEST_TASKS) {
            pool.execute(move|| -> () { panic!() });
        }

        // Ensure new tasks were spawned to compensate.
        let (tx, rx) = channel();
        for _ in range(0, TEST_TASKS) {
            let tx = tx.clone();
            pool.execute(move|| {
                tx.send(1u);
            });
        }

        assert_eq!(rx.iter().take(TEST_TASKS).sum(), TEST_TASKS);
    }

    #[test]
    fn test_should_not_panic_on_drop_if_subtasks_panic_after_drop() {
        use sync::{Arc, Barrier};

        let pool = TaskPool::new(TEST_TASKS);
        let waiter = Arc::new(Barrier::new(TEST_TASKS + 1));

        // Panic all the existing tasks in a bit.
        for _ in range(0, TEST_TASKS) {
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
