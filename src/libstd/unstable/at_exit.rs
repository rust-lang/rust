// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cast;
use libc::size_t;
use rand::RngUtil;
use rand;
use sys;
use task;
use vec;

#[cfg(test)] use uint;

/**
Register a function to be run during runtime shutdown.

After all non-weak tasks have exited, registered exit functions will
execute, in random order, on the primary scheduler. Each function runs
in its own unsupervised task.
*/
pub fn at_exit(f: ~fn()) {
    unsafe {
        let runner: &fn(*ExitFunctions) = exit_runner;
        let runner_pair: sys::Closure = cast::transmute(runner);
        let runner_ptr = runner_pair.code;
        let runner_ptr = cast::transmute(runner_ptr);
        rustrt::rust_register_exit_function(runner_ptr, ~f);
    }
}

// NB: The double pointer indirection here is because ~fn() is a fat
// pointer and due to FFI problems I am more comfortable making the
// interface use a normal pointer
mod rustrt {
    use libc::c_void;

    extern {
        pub fn rust_register_exit_function(runner: *c_void, f: ~~fn());
    }
}

struct ExitFunctions {
    // The number of exit functions
    count: size_t,
    // The buffer of exit functions
    start: *~~fn()
}

fn exit_runner(exit_fns: *ExitFunctions) {
    let exit_fns = unsafe { &*exit_fns };
    let count = (*exit_fns).count;
    let start = (*exit_fns).start;

    // NB: from_buf memcpys from the source, which will
    // give us ownership of the array of functions
    let mut exit_fns_vec = unsafe { vec::from_buf(start, count as uint) };
    // Let's not make any promises about execution order
    let mut rng = rand::rng();
    rng.shuffle_mut(exit_fns_vec);

    debug!("running %u exit functions", exit_fns_vec.len());

    while !exit_fns_vec.is_empty() {
        match exit_fns_vec.pop() {
            ~f => {
                let mut task = task::task();
                task.supervised();
                task.spawn(f);
            }
        }
    }
}

#[test]
fn test_at_exit() {
    let i = 10;
    do at_exit {
        debug!("at_exit1");
        assert_eq!(i, 10);
    }
}

#[test]
fn test_at_exit_many() {
    let i = 10;
    for uint::range(20, 100) |j| {
        do at_exit {
            debug!("at_exit2");
            assert_eq!(i, 10);
            assert!(j > i);
        }
    }
}
