// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[doc(hidden)];

use libc::c_char;

// Some basic logging
macro_rules! rtdebug_ (
    ($( $arg:expr),+) => ( {
        dumb_println(fmt!( $($arg),+ ));

        fn dumb_println(s: &str) {
            use io::WriterUtil;
            let dbg = ::libc::STDERR_FILENO as ::io::fd_t;
            dbg.write_str(s);
            dbg.write_str("\n");
        }

    } )
)

// An alternate version with no output, for turning off logging
macro_rules! rtdebug (
    ($( $arg:expr),+) => ( $(let _ = $arg)*; )
)

mod sched;
mod rtio;
pub mod uvll;
mod uvio;
#[path = "uv/mod.rs"]
mod uv;
#[path = "io/mod.rs"]
mod io;
// FIXME #5248: The import in `sched` doesn't resolve unless this is pub!
pub mod thread_local_storage;
mod work_queue;
mod stack;
mod context;
mod thread;
pub mod env;

#[cfg(stage0)]
pub fn start(main: *u8, _argc: int, _argv: *c_char, _crate_map: *u8) -> int {
    use self::sched::{Scheduler, Task};
    use self::uvio::UvEventLoop;

    let loop_ = ~UvEventLoop::new();
    let mut sched = ~Scheduler::new(loop_);
    let main_task = ~do Task::new(&mut sched.stack_pool) {
        // XXX: Can't call a C function pointer from Rust yet
        unsafe { rust_call_nullary_fn(main) };
    };
    sched.task_queue.push_back(main_task);
    sched.run();
    return 0;

    extern {
        fn rust_call_nullary_fn(f: *u8);
    }
}

#[cfg(not(stage0))]
pub fn start(main: *u8, _argc: int, _argv: **c_char, _crate_map: *u8) -> int {
    use self::sched::{Scheduler, Task};
    use self::uvio::UvEventLoop;

    let loop_ = ~UvEventLoop::new();
    let mut sched = ~Scheduler::new(loop_);
    let main_task = ~do Task::new(&mut sched.stack_pool) {
        // XXX: Can't call a C function pointer from Rust yet
        unsafe { rust_call_nullary_fn(main) };
    };
    sched.task_queue.push_back(main_task);
    sched.run();
    return 0;

    extern {
        fn rust_call_nullary_fn(f: *u8);
    }
}
