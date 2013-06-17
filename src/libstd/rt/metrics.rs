// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use to_str::ToStr;

pub struct SchedMetrics {
    // The number of times executing `run_sched_once`.
    turns: uint,
    // The number of turns that received a message.
    messages_received: uint,
    // The number of turns that ran a task from the queue.
    tasks_resumed_from_queue: uint,
    // The number of turns that found no work to perform.
    wasted_turns: uint,
    // The number of times the scheduler went to sleep.
    sleepy_times: uint,
    // Context switches from the scheduler into a task.
    context_switches_sched_to_task: uint,
    // Context switches from a task into the scheduler.
    context_switches_task_to_sched: uint,
    // Context switches from a task to a task.
    context_switches_task_to_task: uint,
    // Message sends that unblock the receiver
    rendezvous_sends: uint,
    // Message sends that do not unblock the receiver
    non_rendezvous_sends: uint,
    // Message receives that do not block the receiver
    rendezvous_recvs: uint,
    // Message receives that block the receiver
    non_rendezvous_recvs: uint,
    // JoinLatch releases that create tombstones
    release_tombstone: uint,
    // JoinLatch releases that do not create tombstones
    release_no_tombstone: uint,
}

impl SchedMetrics {
    pub fn new() -> SchedMetrics {
        SchedMetrics {
            turns: 0,
            messages_received: 0,
            tasks_resumed_from_queue: 0,
            wasted_turns: 0,
            sleepy_times: 0,
            context_switches_sched_to_task: 0,
            context_switches_task_to_sched: 0,
            context_switches_task_to_task: 0,
            rendezvous_sends: 0,
            non_rendezvous_sends: 0,
            rendezvous_recvs: 0,
            non_rendezvous_recvs: 0,
            release_tombstone: 0,
            release_no_tombstone: 0
        }
    }
}

impl ToStr for SchedMetrics {
    fn to_str(&self) -> ~str {
        fmt!("turns: %u\n\
              messages_received: %u\n\
              tasks_resumed_from_queue: %u\n\
              wasted_turns: %u\n\
              sleepy_times: %u\n\
              context_switches_sched_to_task: %u\n\
              context_switches_task_to_sched: %u\n\
              context_switches_task_to_task: %u\n\
              rendezvous_sends: %u\n\
              non_rendezvous_sends: %u\n\
              rendezvous_recvs: %u\n\
              non_rendezvous_recvs: %u\n\
              release_tombstone: %u\n\
              release_no_tombstone: %u\n\
              ",
             self.turns,
             self.messages_received,
             self.tasks_resumed_from_queue,
             self.wasted_turns,
             self.sleepy_times,
             self.context_switches_sched_to_task,
             self.context_switches_task_to_sched,
             self.context_switches_task_to_task,
             self.rendezvous_sends,
             self.non_rendezvous_sends,
             self.rendezvous_recvs,
             self.non_rendezvous_recvs,
             self.release_tombstone,
             self.release_no_tombstone
        )
    }
}