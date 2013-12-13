// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Homing I/O implementation
//!
//! In libuv, whenever a handle is created on an I/O loop it is illegal to use
//! that handle outside of that I/O loop. We use libuv I/O with our green
//! scheduler, and each green scheduler corresponds to a different I/O loop on a
//! different OS thread. Green tasks are also free to roam among schedulers,
//! which implies that it is possible to create an I/O handle on one event loop
//! and then attempt to use it on another.
//!
//! In order to solve this problem, this module implements the notion of a
//! "homing operation" which will transplant a task from its currently running
//! scheduler back onto the original I/O loop. This is accomplished entirely at
//! the librustuv layer with very little cooperation from the scheduler (which
//! we don't even know exists technically).
//!
//! These homing operations are completed by first realizing that we're on the
//! wrong I/O loop, then descheduling ourselves, sending ourselves to the
//! correct I/O loop, and then waking up the I/O loop in order to process its
//! local queue of tasks which need to run.
//!
//! This enqueueing is done with a concurrent queue from libstd, and the
//! signalling is achieved with an async handle.

use std::rt::local::Local;
use std::rt::rtio::LocalIo;
use std::rt::task::{Task, BlockedTask};

use ForbidUnwind;
use queue::{Queue, QueuePool};

/// A handle to a remote libuv event loop. This handle will keep the event loop
/// alive while active in order to ensure that a homing operation can always be
/// completed.
///
/// Handles are clone-able in order to derive new handles from existing handles
/// (very useful for when accepting a socket from a server).
pub struct HomeHandle {
    priv queue: Queue,
    priv id: uint,
}

impl HomeHandle {
    pub fn new(id: uint, pool: &mut QueuePool) -> HomeHandle {
        HomeHandle { queue: pool.queue(), id: id }
    }

    fn send(&mut self, task: BlockedTask) {
        self.queue.push(task);
    }
}

impl Clone for HomeHandle {
    fn clone(&self) -> HomeHandle {
        HomeHandle {
            queue: self.queue.clone(),
            id: self.id,
        }
    }
}

pub trait HomingIO {
    fn home<'r>(&'r mut self) -> &'r mut HomeHandle;

    /// This function will move tasks to run on their home I/O scheduler. Note
    /// that this function does *not* pin the task to the I/O scheduler, but
    /// rather it simply moves it to running on the I/O scheduler.
    fn go_to_IO_home(&mut self) -> uint {
        let _f = ForbidUnwind::new("going home");

        let mut cur_task: ~Task = Local::take();
        let cur_loop_id = {
            let mut io = cur_task.local_io().expect("libuv must have I/O");
            io.get().id()
        };

        // Try at all costs to avoid the homing operation because it is quite
        // expensive. Hence, we only deschedule/send if we're not on the correct
        // event loop. If we're already on the home event loop, then we're good
        // to go (remember we have no preemption, so we're guaranteed to stay on
        // this event loop as long as we avoid the scheduler).
        if cur_loop_id != self.home().id {
            cur_task.deschedule(1, |task| {
                self.home().send(task);
                Ok(())
            });

            // Once we wake up, assert that we're in the right location
            let cur_loop_id = {
                let mut io = LocalIo::borrow().expect("libuv must have I/O");
                io.get().id()
            };
            assert_eq!(cur_loop_id, self.home().id);

            cur_loop_id
        } else {
            Local::put(cur_task);
            cur_loop_id
        }
    }

    /// Fires a single homing missile, returning another missile targeted back
    /// at the original home of this task. In other words, this function will
    /// move the local task to its I/O scheduler and then return an RAII wrapper
    /// which will return the task home.
    fn fire_homing_missile(&mut self) -> HomingMissile {
        HomingMissile { io_home: self.go_to_IO_home() }
    }
}

/// After a homing operation has been completed, this will return the current
/// task back to its appropriate home (if applicable). The field is used to
/// assert that we are where we think we are.
struct HomingMissile {
    priv io_home: uint,
}

impl HomingMissile {
    /// Check at runtime that the task has *not* transplanted itself to a
    /// different I/O loop while executing.
    pub fn check(&self, msg: &'static str) {
        let mut io = LocalIo::borrow().expect("libuv must have I/O");
        assert!(io.get().id() == self.io_home, "{}", msg);
    }
}

impl Drop for HomingMissile {
    fn drop(&mut self) {
        let _f = ForbidUnwind::new("leaving home");

        // It would truly be a sad day if we had moved off the home I/O
        // scheduler while we were doing I/O.
        self.check("task moved away from the home scheduler");
    }
}

#[cfg(test)]
mod test {
    // On one thread, create a udp socket. Then send that socket to another
    // thread and destroy the socket on the remote thread. This should make sure
    // that homing kicks in for the socket to go back home to the original
    // thread, close itself, and then come back to the last thread.
    //#[test]
    //fn test_homing_closes_correctly() {
    //    let (port, chan) = Chan::new();

    //    do task::spawn_sched(task::SingleThreaded) {
    //        let listener = UdpWatcher::bind(local_loop(), next_test_ip4()).unwrap();
    //        chan.send(listener);
    //    }

    //    do task::spawn_sched(task::SingleThreaded) {
    //        port.recv();
    //    }
    //}

    // This is a bit of a crufty old test, but it has its uses.
    //#[test]
    //fn test_simple_homed_udp_io_bind_then_move_task_then_home_and_close() {
    //    use std::cast;
    //    use std::rt::local::Local;
    //    use std::rt::rtio::{EventLoop, IoFactory};
    //    use std::rt::sched::Scheduler;
    //    use std::rt::sched::{Shutdown, TaskFromFriend};
    //    use std::rt::sleeper_list::SleeperList;
    //    use std::rt::task::Task;
    //    use std::rt::task::UnwindResult;
    //    use std::rt::thread::Thread;
    //    use std::rt::deque::BufferPool;
    //    use std::unstable::run_in_bare_thread;
    //    use uvio::UvEventLoop;

    //    do run_in_bare_thread {
    //        let sleepers = SleeperList::new();
    //        let mut pool = BufferPool::new();
    //        let (worker1, stealer1) = pool.deque();
    //        let (worker2, stealer2) = pool.deque();
    //        let queues = ~[stealer1, stealer2];

    //        let loop1 = ~UvEventLoop::new() as ~EventLoop;
    //        let mut sched1 = ~Scheduler::new(loop1, worker1, queues.clone(),
    //                                         sleepers.clone());
    //        let loop2 = ~UvEventLoop::new() as ~EventLoop;
    //        let mut sched2 = ~Scheduler::new(loop2, worker2, queues.clone(),
    //                                         sleepers.clone());

    //        let handle1 = sched1.make_handle();
    //        let handle2 = sched2.make_handle();
    //        let tasksFriendHandle = sched2.make_handle();

    //        let on_exit: proc(UnwindResult) = proc(exit_status) {
    //            let mut handle1 = handle1;
    //            let mut handle2 = handle2;
    //            handle1.send(Shutdown);
    //            handle2.send(Shutdown);
    //            assert!(exit_status.is_success());
    //        };

    //        unsafe fn local_io() -> &'static mut IoFactory {
    //            let mut sched = Local::borrow(None::<Scheduler>);
    //            let io = sched.get().event_loop.io();
    //            cast::transmute(io.unwrap())
    //        }

    //        let test_function: proc() = proc() {
    //            let io = unsafe { local_io() };
    //            let addr = next_test_ip4();
    //            let maybe_socket = io.udp_bind(addr);
    //            // this socket is bound to this event loop
    //            assert!(maybe_socket.is_ok());

    //            // block self on sched1
    //            let scheduler: ~Scheduler = Local::take();
    //            let mut tasksFriendHandle = Some(tasksFriendHandle);
    //            scheduler.deschedule_running_task_and_then(|_, task| {
    //                // unblock task
    //                task.wake().map(|task| {
    //                    // send self to sched2
    //                    tasksFriendHandle.take_unwrap()
    //                                     .send(TaskFromFriend(task));
    //                });
    //                // sched1 should now sleep since it has nothing else to do
    //            })
    //            // sched2 will wake up and get the task as we do nothing else,
    //            // the function ends and the socket goes out of scope sched2
    //            // will start to run the destructor the destructor will first
    //            // block the task, set it's home as sched1, then enqueue it
    //            // sched2 will dequeue the task, see that it has a home, and
    //            // send it to sched1 sched1 will wake up, exec the close
    //            // function on the correct loop, and then we're done
    //        };

    //        let mut main_task = ~Task::new_root(&mut sched1.stack_pool, None,
    //                                            test_function);
    //        main_task.death.on_exit = Some(on_exit);

    //        let null_task = ~do Task::new_root(&mut sched2.stack_pool, None) {
    //            // nothing
    //        };

    //        let main_task = main_task;
    //        let sched1 = sched1;
    //        let thread1 = do Thread::start {
    //            sched1.bootstrap(main_task);
    //        };

    //        let sched2 = sched2;
    //        let thread2 = do Thread::start {
    //            sched2.bootstrap(null_task);
    //        };

    //        thread1.join();
    //        thread2.join();
    //    }
    //}
}
