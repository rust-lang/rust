// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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

/// Get a port number, starting at 9600, for use in tests
pub fn next_test_port() -> u16 {
    unsafe {
        return rust_dbg_next_port() as u16;
    }
    extern {
        fn rust_dbg_next_port() -> ::libc::uintptr_t;
    }
}
