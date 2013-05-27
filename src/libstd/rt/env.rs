// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Runtime environment settings

use libc::{size_t, c_char, c_int};

pub struct Environment {
    /// The number of threads to use by default
    num_sched_threads: size_t,
    /// The minimum size of a stack segment
    min_stack_size: size_t,
    /// The maximum amount of total stack per task before aborting
    max_stack_size: size_t,
    /// The default logging configuration
    logspec: *c_char,
    /// Record and report detailed information about memory leaks
    detailed_leaks: bool,
    /// Seed the random number generator
    rust_seed: *c_char,
    /// Poison allocations on free
    poison_on_free: bool,
    /// The argc value passed to main
    argc: c_int,
    /// The argv value passed to main
    argv: **c_char,
    /// Print GC debugging info (true if env var RUST_DEBUG_MEM is set)
    debug_mem: bool,
    /// Print GC debugging info (true if env var RUST_DEBUG_BORROW is set)
    debug_borrow: bool,
}

/// Get the global environment settings
/// # Safety Note
/// This will abort the process if run outside of task context
pub fn get() -> &Environment {
    unsafe { rust_get_rt_env() }
}

extern {
    fn rust_get_rt_env() -> &Environment;
}
