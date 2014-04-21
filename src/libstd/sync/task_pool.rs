// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
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

use task;
use task::spawn;
use vec::Vec;
use comm::{channel, Sender};

enum Msg<T> {
    Execute(proc(&T):Send),
    Quit
}

/// A task pool used to execute functions in parallel.
pub struct TaskPool<T> {
    channels: Vec<Sender<Msg<T>>>,
    next_index: uint,
}

#[unsafe_destructor]
impl<T> Drop for TaskPool<T> {
    fn drop(&mut self) {
        for channel in self.channels.mut_iter() {
            channel.send(Quit);
        }
    }
}

impl<T> TaskPool<T> {
    /// Spawns a new task pool with `n_tasks` tasks. The provided
    /// `init_fn_factory` returns a function which, given the index of the
    /// task, should return local data to be kept around in that task.
    ///
    /// # Failure
    ///
    /// This function will fail if `n_tasks` is less than 1.
    pub fn new(n_tasks: uint,
               init_fn_factory: || -> proc(uint):Send -> T)
               -> TaskPool<T> {
        assert!(n_tasks >= 1);

        let channels = Vec::from_fn(n_tasks, |i| {
            let (tx, rx) = channel::<Msg<T>>();
            let init_fn = init_fn_factory();

            let task_body = proc() {
                let local_data = init_fn(i);
                loop {
                    match rx.recv() {
                        Execute(f) => f(&local_data),
                        Quit => break
                    }
                }
            };

            // Run on this scheduler.
            task::spawn(task_body);

            tx
        });

        return TaskPool {
            channels: channels,
            next_index: 0,
        };
    }

    /// Executes the function `f` on a task in the pool. The function
    /// receives a reference to the local data returned by the `init_fn`.
    pub fn execute(&mut self, f: proc(&T):Send) {
        self.channels.get(self.next_index).send(Execute(f));
        self.next_index += 1;
        if self.next_index == self.channels.len() { self.next_index = 0; }
    }
}

#[test]
fn test_task_pool() {
    let f: || -> proc(uint):Send -> uint = || { proc(i) i };
    let mut pool = TaskPool::new(4, f);
    for _ in range(0u, 8) {
        pool.execute(proc(i) println!("Hello from thread {}!", *i));
    }
}

#[test]
#[should_fail]
fn test_zero_tasks_failure() {
    let f: || -> proc(uint):Send -> uint = || { proc(i) i };
    TaskPool::new(0, f);
}
