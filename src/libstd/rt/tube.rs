// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A very simple unsynchronized channel type for sending buffered data from
//! scheduler context to task context.
//!
//! XXX: This would be safer to use if split into two types like Port/Chan

use option::*;
use clone::Clone;
use super::rc::RC;
use rt::sched::Scheduler;
use rt::{context, TaskContext, SchedulerContext};
use rt::kill::BlockedTask;
use rt::local::Local;
use vec::OwnedVector;
use container::Container;

struct TubeState<T> {
    blocked_task: Option<BlockedTask>,
    buf: ~[T]
}

pub struct Tube<T> {
    p: RC<TubeState<T>>
}

impl<T> Tube<T> {
    pub fn new() -> Tube<T> {
        Tube {
            p: RC::new(TubeState {
                blocked_task: None,
                buf: ~[]
            })
        }
    }

    pub fn send(&mut self, val: T) {
        rtdebug!("tube send");
        assert!(context() == SchedulerContext);

        unsafe {
            let state = self.p.unsafe_borrow_mut();
            (*state).buf.push(val);

            if (*state).blocked_task.is_some() {
                // There's a waiting task. Wake it up
                rtdebug!("waking blocked tube");
                let task = (*state).blocked_task.take_unwrap();
                let sched = Local::take::<Scheduler>();
                sched.resume_blocked_task_immediately(task);
            }
        }
    }

    pub fn recv(&mut self) -> T {
        assert!(context() == TaskContext);

        unsafe {
            let state = self.p.unsafe_borrow_mut();
            if !(*state).buf.is_empty() {
                return (*state).buf.shift();
            } else {
                // Block and wait for the next message
                rtdebug!("blocking on tube recv");
                assert!(self.p.refcount() > 1); // There better be somebody to wake us up
                assert!((*state).blocked_task.is_none());
                let sched = Local::take::<Scheduler>();
                do sched.deschedule_running_task_and_then |_, task| {
                    (*state).blocked_task = Some(task);
                }
                rtdebug!("waking after tube recv");
                let buf = &mut (*state).buf;
                assert!(!buf.is_empty());
                return buf.shift();
            }
        }
    }
}

impl<T> Clone for Tube<T> {
    fn clone(&self) -> Tube<T> {
        Tube { p: self.p.clone() }
    }
}

#[cfg(test)]
mod test {
    use int;
    use cell::Cell;
    use rt::test::*;
    use rt::rtio::EventLoop;
    use rt::sched::Scheduler;
    use rt::local::Local;
    use super::*;

    #[test]
    fn simple_test() {
        do run_in_newsched_task {
            let mut tube: Tube<int> = Tube::new();
            let tube_clone = tube.clone();
            let tube_clone_cell = Cell::new(tube_clone);
            let sched = Local::take::<Scheduler>();
            do sched.deschedule_running_task_and_then |sched, task| {
                let mut tube_clone = tube_clone_cell.take();
                tube_clone.send(1);
                sched.enqueue_blocked_task(task);
            }

            assert!(tube.recv() == 1);
        }
    }

    #[test]
    fn blocking_test() {
        do run_in_newsched_task {
            let mut tube: Tube<int> = Tube::new();
            let tube_clone = tube.clone();
            let tube_clone = Cell::new(tube_clone);
            let sched = Local::take::<Scheduler>();
            do sched.deschedule_running_task_and_then |sched, task| {
                let tube_clone = Cell::new(tube_clone.take());
                do sched.event_loop.callback {
                    let mut tube_clone = tube_clone.take();
                    // The task should be blocked on this now and
                    // sending will wake it up.
                    tube_clone.send(1);
                }
                sched.enqueue_blocked_task(task);
            }

            assert!(tube.recv() == 1);
        }
    }

    #[test]
    fn many_blocking_test() {
        static MAX: int = 100;

        do run_in_newsched_task {
            let mut tube: Tube<int> = Tube::new();
            let tube_clone = tube.clone();
            let tube_clone = Cell::new(tube_clone);
            let sched = Local::take::<Scheduler>();
            do sched.deschedule_running_task_and_then |sched, task| {
                callback_send(tube_clone.take(), 0);

                fn callback_send(tube: Tube<int>, i: int) {
                    if i == 100 { return; }

                    let tube = Cell::new(Cell::new(tube));
                    do Local::borrow::<Scheduler, ()> |sched| {
                        let tube = tube.take();
                        do sched.event_loop.callback {
                            let mut tube = tube.take();
                            // The task should be blocked on this now and
                            // sending will wake it up.
                            tube.send(i);
                            callback_send(tube, i + 1);
                        }
                    }
                }

                sched.enqueue_blocked_task(task);
            }

            for int::range(0, MAX) |i| {
                let j = tube.recv();
                assert!(j == i);
            }
        }
    }
}
