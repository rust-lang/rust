// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::*;
// use either::{Either, Left, Right};
use rt::kill::BlockedTask;
use rt::sched::Scheduler;
use rt::local::Local;

/// Trait for message-passing primitives that can be select()ed on.
pub trait Select {
    // Returns true if data was available.
    fn optimistic_check(&mut self) -> bool;
    // Returns true if data was available. If so, shall also wake() the task.
    fn block_on(&mut self, &mut Scheduler, BlockedTask) -> bool;
    // Returns true if data was available.
    fn unblock_from(&mut self) -> bool;
}

/// Trait for message-passing primitives that can use the select2() convenience wrapper.
// (This is separate from the above trait to enable heterogeneous lists of ports
// that implement Select on different types to use select().)
pub trait SelectPort<T> : Select {
    fn recv_ready(self) -> Option<T>;
}

/// Receive a message from any one of many ports at once.
pub fn select<A: Select>(ports: &mut [A]) -> uint {
    if ports.is_empty() {
        fail!("can't select on an empty list");
    }

    for ports.mut_iter().enumerate().advance |(index, port)| {
        if port.optimistic_check() {
            return index;
        }
    }

    // If one of the ports already contains data when we go to block on it, we
    // don't bother enqueueing on the rest of them, so we shouldn't bother
    // unblocking from it either. This is just for efficiency, not correctness.
    // (If not, we need to unblock from all of them. Length is a placeholder.)
    let mut ready_index = ports.len();

    let sched = Local::take::<Scheduler>();
    do sched.deschedule_running_task_and_then |sched, task| {
        let task_handles = task.make_selectable(ports.len());

        for ports.mut_iter().zip(task_handles.consume_iter()).enumerate().advance
                |(index, (port, task_handle))| {
            // If one of the ports has data by now, it will wake the handle.
            if port.block_on(sched, task_handle) {
                ready_index = index;
                break;
            }
        }
    }

    // Task resumes. Now unblock ourselves from all the ports we blocked on.
    // If the success index wasn't reset, 'take' will just take all of them.
    // Iterate in reverse so the 'earliest' index that's ready gets returned.
    for ports.mut_slice(0, ready_index).mut_rev_iter().enumerate().advance |(index, port)| {
        if port.unblock_from() {
            ready_index = index;
        }
    }

    assert!(ready_index < ports.len());
    return ready_index;
}

/* FIXME(#5121, #7914) This all should be legal, but rust is not clever enough yet.

impl <'self> Select for &'self mut Select {
    fn optimistic_check(&mut self) -> bool { self.optimistic_check() }
    fn block_on(&mut self, sched: &mut Scheduler, task: BlockedTask) -> bool {
        self.block_on(sched, task)
    }
    fn unblock_from(&mut self) -> bool { self.unblock_from() }
}

pub fn select2<TA, A: SelectPort<TA>, TB, B: SelectPort<TB>>(mut a: A, mut b: B)
        -> Either<(Option<TA>, B), (A, Option<TB>)> {
    let result = {
        let mut ports = [&mut a as &mut Select, &mut b as &mut Select];
        select(ports)
    };
    match result {
        0 => Left ((a.recv_ready(), b)),
        1 => Right((a, b.recv_ready())),
        x => fail!("impossible case in select2: %?", x)
    }
}

*/
