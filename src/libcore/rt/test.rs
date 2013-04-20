// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::io::net::ip::{IpAddr, Ipv4};

// For setting up tests of the new scheduler
pub fn run_in_newsched_task(f: ~fn()) {
    use cell::Cell;
    use unstable::run_in_bare_thread;
    use super::sched::Task;
    use super::uvio::UvEventLoop;

    let f = Cell(Cell(f));

    do run_in_bare_thread {
        let mut sched = ~UvEventLoop::new_scheduler();
        let f = f.take();
        let task = ~do Task::new(&mut sched.stack_pool) {
            (f.take())();
        };
        sched.task_queue.push_back(task);
        sched.run();
    }
}

/// Create a new task and run it right now
pub fn spawn_immediately(f: ~fn()) {
    use cell::Cell;
    use super::*;
    use super::sched::*;

    let mut sched = local_sched::take();
    let task = ~Task::new(&mut sched.stack_pool, f);
    do sched.switch_running_tasks_and_then(task) |task| {
        let task = Cell(task);
        do local_sched::borrow |sched| {
            sched.task_queue.push_front(task.take());
        }
    }
}

/// Get a port number, starting at 9600, for use in tests
pub fn next_test_port() -> u16 {
    unsafe {
        return rust_dbg_next_port() as u16;
    }
    extern {
        fn rust_dbg_next_port() -> ::libc::uintptr_t;
    }
}

/// Get a unique localhost:port pair starting at 9600
pub fn next_test_ip4() -> IpAddr {
    Ipv4(127, 0, 0, 1, next_test_port())
}
