// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!**************************************************************************
 *
 * WARNING: linked failure has been removed since this doc comment was written,
 *          but it was so pretty that I didn't want to remove it.
 *
 * Spawning & linked failure
 *
 * Several data structures are involved in task management to allow properly
 * propagating failure across linked/supervised tasks.
 *
 * (1) The "taskgroup_arc" is an unsafe::exclusive which contains a hashset of
 *     all tasks that are part of the group. Some tasks are 'members', which
 *     means if they fail, they will kill everybody else in the taskgroup.
 *     Other tasks are 'descendants', which means they will not kill tasks
 *     from this group, but can be killed by failing members.
 *
 *     A new one of these is created each spawn_linked or spawn_supervised.
 *
 * (2) The "taskgroup" is a per-task control structure that tracks a task's
 *     spawn configuration. It contains a reference to its taskgroup_arc, a
 *     reference to its node in the ancestor list (below), and an optionally
 *     configured notification port. These are stored in TLS.
 *
 * (3) The "ancestor_list" is a cons-style list of unsafe::exclusives which
 *     tracks 'generations' of taskgroups -- a group's ancestors are groups
 *     which (directly or transitively) spawn_supervised-ed them. Each task
 *     is recorded in the 'descendants' of each of its ancestor groups.
 *
 *     Spawning a supervised task is O(n) in the number of generations still
 *     alive, and exiting (by success or failure) that task is also O(n).
 *
 * This diagram depicts the references between these data structures:
 *
 *          linked_________________________________
 *        ___/                   _________         \___
 *       /   \                  | group X |        /   \
 *      (  A  ) - - - - - - - > | {A,B} {}|< - - -(  B  )
 *       \___/                  |_________|        \___/
 *      unlinked
 *         |      __ (nil)
 *         |      //|                         The following code causes this:
 *         |__   //   /\         _________
 *        /   \ //    ||        | group Y |     fn taskA() {
 *       (  C  )- - - ||- - - > |{C} {D,E}|         spawn(taskB);
 *        \___/      /  \=====> |_________|         spawn_unlinked(taskC);
 *      supervise   /gen \                          ...
 *         |    __  \ 00 /                      }
 *         |    //|  \__/                       fn taskB() { ... }
 *         |__ //     /\         _________      fn taskC() {
 *        /   \/      ||        | group Z |         spawn_supervised(taskD);
 *       (  D  )- - - ||- - - > | {D} {E} |         ...
 *        \___/      /  \=====> |_________|     }
 *      supervise   /gen \                      fn taskD() {
 *         |    __  \ 01 /                          spawn_supervised(taskE);
 *         |    //|  \__/                           ...
 *         |__ //                _________      }
 *        /   \/                | group W |     fn taskE() { ... }
 *       (  E  )- - - - - - - > | {E}  {} |
 *        \___/                 |_________|
 *
 *        "tcb"               "taskgroup_arc"
 *             "ancestor_list"
 *
 ****************************************************************************/

#[doc(hidden)];

use prelude::*;

use cell::Cell;
use comm::{GenericChan, oneshot};
use rt::local::Local;
use rt::sched::{Scheduler, Shutdown, TaskFromFriend};
use rt::task::{Task, Sched};
use rt::task::UnwindResult;
use rt::thread::Thread;
use rt::work_queue::WorkQueue;
use rt::{in_green_task_context, new_event_loop};
use task::SingleThreaded;
use task::TaskOpts;

#[cfg(test)] use task::default_task_opts;
#[cfg(test)] use comm;
#[cfg(test)] use task;

pub fn spawn_raw(mut opts: TaskOpts, f: proc()) {
    assert!(in_green_task_context());

    let mut task = if opts.sched.mode != SingleThreaded {
        if opts.watched {
            Task::build_child(opts.stack_size, f)
        } else {
            Task::build_root(opts.stack_size, f)
        }
    } else {
        unsafe {
            // Creating a 1:1 task:thread ...
            let sched: *mut Scheduler = Local::unsafe_borrow();
            let sched_handle = (*sched).make_handle();

            // Since this is a 1:1 scheduler we create a queue not in
            // the stealee set. The run_anything flag is set false
            // which will disable stealing.
            let work_queue = WorkQueue::new();

            // Create a new scheduler to hold the new task
            let mut new_sched = ~Scheduler::new_special(new_event_loop(),
                                                        work_queue,
                                                        (*sched).work_queues.clone(),
                                                        (*sched).sleeper_list.clone(),
                                                        false,
                                                        Some(sched_handle));
            let mut new_sched_handle = new_sched.make_handle();

            // Allow the scheduler to exit when the pinned task exits
            new_sched_handle.send(Shutdown);

            // Pin the new task to the new scheduler
            let new_task = if opts.watched {
                Task::build_homed_child(opts.stack_size, f, Sched(new_sched_handle))
            } else {
                Task::build_homed_root(opts.stack_size, f, Sched(new_sched_handle))
            };

            // Create a task that will later be used to join with the new scheduler
            // thread when it is ready to terminate
            let (thread_port, thread_chan) = oneshot();
            let thread_port_cell = Cell::new(thread_port);
            let join_task = do Task::build_child(None) {
                debug!("running join task");
                let thread_port = thread_port_cell.take();
                let thread: Thread = thread_port.recv();
                thread.join();
            };

            // Put the scheduler into another thread
            let new_sched_cell = Cell::new(new_sched);
            let orig_sched_handle_cell = Cell::new((*sched).make_handle());
            let join_task_cell = Cell::new(join_task);

            let thread = do Thread::start {
                let mut new_sched = new_sched_cell.take();
                let mut orig_sched_handle = orig_sched_handle_cell.take();
                let join_task = join_task_cell.take();

                let bootstrap_task = ~do Task::new_root(&mut new_sched.stack_pool, None) || {
                    debug!("boostrapping a 1:1 scheduler");
                };
                new_sched.bootstrap(bootstrap_task);

                // Now tell the original scheduler to join with this thread
                // by scheduling a thread-joining task on the original scheduler
                orig_sched_handle.send(TaskFromFriend(join_task));

                // NB: We can't simply send a message from here to another task
                // because this code isn't running in a task and message passing doesn't
                // work outside of tasks. Hence we're sending a scheduler message
                // to execute a new task directly to a scheduler.
            };

            // Give the thread handle to the join task
            thread_chan.send(thread);

            // When this task is enqueued on the current scheduler it will then get
            // forwarded to the scheduler to which it is pinned
            new_task
        }
    };

    if opts.notify_chan.is_some() {
        let notify_chan = opts.notify_chan.take_unwrap();
        let notify_chan = Cell::new(notify_chan);
        let on_exit: proc(UnwindResult) = |task_result| {
            notify_chan.take().send(task_result)
        };
        task.death.on_exit = Some(on_exit);
    }

    task.name = opts.name.take();
    debug!("spawn calling run_task");
    Scheduler::run_task(task);

}

#[test]
fn test_spawn_raw_simple() {
    let (po, ch) = stream();
    do spawn_raw(default_task_opts()) {
        ch.send(());
    }
    po.recv();
}

#[test]
fn test_spawn_raw_unsupervise() {
    let opts = task::TaskOpts {
        watched: false,
        notify_chan: None,
        .. default_task_opts()
    };
    do spawn_raw(opts) {
        fail!();
    }
}

#[test]
fn test_spawn_raw_notify_success() {
    let (notify_po, notify_ch) = comm::stream();

    let opts = task::TaskOpts {
        notify_chan: Some(notify_ch),
        .. default_task_opts()
    };
    do spawn_raw(opts) {
    }
    assert!(notify_po.recv().is_success());
}

#[test]
fn test_spawn_raw_notify_failure() {
    // New bindings for these
    let (notify_po, notify_ch) = comm::stream();

    let opts = task::TaskOpts {
        watched: false,
        notify_chan: Some(notify_ch),
        .. default_task_opts()
    };
    do spawn_raw(opts) {
        fail!();
    }
    assert!(notify_po.recv().is_failure());
}
