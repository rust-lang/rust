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

#[allow(dead_code)];

use std::cast;
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

pub fn local_id() -> uint {
    let mut io = match LocalIo::borrow() {
        Some(io) => io, None => return 0,
    };
    let io = io.get();
    unsafe {
        let (_vtable, ptr): (uint, uint) = cast::transmute(io);
        return ptr;
    }
}

pub trait HomingIO {
    fn home<'r>(&'r mut self) -> &'r mut HomeHandle;

    /// This function will move tasks to run on their home I/O scheduler. Note
    /// that this function does *not* pin the task to the I/O scheduler, but
    /// rather it simply moves it to running on the I/O scheduler.
    fn go_to_IO_home(&mut self) -> uint {
        let _f = ForbidUnwind::new("going home");

        let cur_loop_id = local_id();
        let destination = self.home().id;

        // Try at all costs to avoid the homing operation because it is quite
        // expensive. Hence, we only deschedule/send if we're not on the correct
        // event loop. If we're already on the home event loop, then we're good
        // to go (remember we have no preemption, so we're guaranteed to stay on
        // this event loop as long as we avoid the scheduler).
        if cur_loop_id != destination {
            let cur_task: ~Task = Local::take();
            cur_task.deschedule(1, |task| {
                self.home().send(task);
                Ok(())
            });

            // Once we wake up, assert that we're in the right location
            assert_eq!(local_id(), destination);
        }

        return destination;
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
    io_home: uint,
}

impl HomingMissile {
    /// Check at runtime that the task has *not* transplanted itself to a
    /// different I/O loop while executing.
    pub fn check(&self, msg: &'static str) {
        assert!(local_id() == self.io_home, "{}", msg);
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
    use green::sched;
    use green::{SchedPool, PoolConfig};
    use std::rt::rtio::RtioUdpSocket;
    use std::io::test::next_test_ip4;
    use std::task::TaskOpts;

    use net::UdpWatcher;
    use super::super::local_loop;

    // On one thread, create a udp socket. Then send that socket to another
    // thread and destroy the socket on the remote thread. This should make sure
    // that homing kicks in for the socket to go back home to the original
    // thread, close itself, and then come back to the last thread.
    #[test]
    fn test_homing_closes_correctly() {
        let (port, chan) = Chan::new();
        let mut pool = SchedPool::new(PoolConfig {
            threads: 1,
            event_loop_factory: None,
        });

        pool.spawn(TaskOpts::new(), proc() {
            let listener = UdpWatcher::bind(local_loop(), next_test_ip4());
            chan.send(listener.unwrap());
        });

        let task = pool.task(TaskOpts::new(), proc() {
            drop(port.recv());
        });
        pool.spawn_sched().send(sched::TaskFromFriend(task));

        pool.shutdown();
    }

    #[test]
    fn test_homing_read() {
        let (port, chan) = Chan::new();
        let mut pool = SchedPool::new(PoolConfig {
            threads: 1,
            event_loop_factory: None,
        });

        pool.spawn(TaskOpts::new(), proc() {
            let addr1 = next_test_ip4();
            let addr2 = next_test_ip4();
            let listener = UdpWatcher::bind(local_loop(), addr2);
            chan.send((listener.unwrap(), addr1));
            let mut listener = UdpWatcher::bind(local_loop(), addr1).unwrap();
            listener.sendto([1, 2, 3, 4], addr2).unwrap();
        });

        let task = pool.task(TaskOpts::new(), proc() {
            let (mut watcher, addr) = port.recv();
            let mut buf = [0, ..10];
            assert_eq!(watcher.recvfrom(buf).unwrap(), (4, addr));
        });
        pool.spawn_sched().send(sched::TaskFromFriend(task));

        pool.shutdown();
    }
}
