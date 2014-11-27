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

use task::spawn;
use comm::{channel, Sender, Receiver};
use sync::{Arc, Mutex};

struct Sentinel<'a> {
    jobs: &'a Arc<Mutex<Receiver<proc(): Send>>>,
    active: bool
}

impl<'a> Sentinel<'a> {
    fn new(jobs: &Arc<Mutex<Receiver<proc(): Send>>>) -> Sentinel { unimplemented!() }

    // Cancel and destroy this sentinel.
    fn cancel(mut self) { unimplemented!() }
}

#[unsafe_destructor]
impl<'a> Drop for Sentinel<'a> {
    fn drop(&mut self) { unimplemented!() }
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
///     pool.execute(proc() {
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
    jobs: Sender<proc(): Send>
}

impl TaskPool {
    /// Spawns a new task pool with `tasks` tasks.
    ///
    /// # Panics
    ///
    /// This function will panic if `tasks` is 0.
    pub fn new(tasks: uint) -> TaskPool { unimplemented!() }

    /// Executes the function `job` on a task in the pool.
    pub fn execute(&self, job: proc():Send) { unimplemented!() }
}

fn spawn_in_pool(jobs: Arc<Mutex<Receiver<proc(): Send>>>) { unimplemented!() }

