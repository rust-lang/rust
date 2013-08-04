// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

/// A task pool abstraction. Useful for achieving predictable CPU
/// parallelism.


use std::comm::Chan;
use std::comm;
use std::task::SchedMode;
use std::task;
use std::vec;

#[cfg(test)] use std::task::SingleThreaded;

enum Msg<T> {
    Execute(~fn(&T)),
    Quit
}

pub struct TaskPool<T> {
    channels: ~[Chan<Msg<T>>],
    next_index: uint,
}

#[unsafe_destructor]
impl<T> Drop for TaskPool<T> {
    fn drop(&self) {
        for channel in self.channels.iter() {
            channel.send(Quit);
        }
    }
}

impl<T> TaskPool<T> {
    /// Spawns a new task pool with `n_tasks` tasks. If the `sched_mode`
    /// is None, the tasks run on this scheduler; otherwise, they run on a
    /// new scheduler with the given mode. The provided `init_fn_factory`
    /// returns a function which, given the index of the task, should return
    /// local data to be kept around in that task.
    pub fn new(n_tasks: uint,
               opt_sched_mode: Option<SchedMode>,
               init_fn_factory: ~fn() -> ~fn(uint) -> T)
               -> TaskPool<T> {
        assert!(n_tasks >= 1);

        let channels = do vec::from_fn(n_tasks) |i| {
            let (port, chan) = comm::stream::<Msg<T>>();
            let init_fn = init_fn_factory();

            let task_body: ~fn() = || {
                let local_data = init_fn(i);
                loop {
                    match port.recv() {
                        Execute(f) => f(&local_data),
                        Quit => break
                    }
                }
            };

            // Start the task.
            match opt_sched_mode {
                None => {
                    // Run on this scheduler.
                    task::spawn(task_body);
                }
                Some(sched_mode) => {
                    let mut task = task::task();
                    task.sched_mode(sched_mode);
                    task.spawn(task_body);
                }
            }

            chan
        };

        return TaskPool { channels: channels, next_index: 0 };
    }

    /// Executes the function `f` on a task in the pool. The function
    /// receives a reference to the local data returned by the `init_fn`.
    pub fn execute(&mut self, f: ~fn(&T)) {
        self.channels[self.next_index].send(Execute(f));
        self.next_index += 1;
        if self.next_index == self.channels.len() { self.next_index = 0; }
    }
}

#[test]
fn test_task_pool() {
    let f: ~fn() -> ~fn(uint) -> uint = || {
        let g: ~fn(uint) -> uint = |i| i;
        g
    };
    let mut pool = TaskPool::new(4, Some(SingleThreaded), f);
    do 8.times {
        pool.execute(|i| printfln!("Hello from thread %u!", *i));
    }
}
