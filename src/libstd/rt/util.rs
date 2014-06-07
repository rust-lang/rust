// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use from_str::FromStr;
use from_str::from_str;
use libc::uintptr_t;
use libc;
use option::{Some, None, Option};
use os;
use str::Str;
use sync::atomics;

/// Get the number of cores available
pub fn num_cpus() -> uint {
    unsafe {
        return rust_get_num_cpus();
    }

    extern {
        fn rust_get_num_cpus() -> libc::uintptr_t;
    }
}

/// Dynamically inquire about whether we're running under V.
/// You should usually not use this unless your test definitely
/// can't run correctly un-altered. Valgrind is there to help
/// you notice weirdness in normal, un-doctored code paths!
pub fn running_on_valgrind() -> bool {
    extern {
        fn rust_running_on_valgrind() -> uintptr_t;
    }
    unsafe { rust_running_on_valgrind() != 0 }
}

/// Valgrind has a fixed-sized array (size around 2000) of segment descriptors
/// wired into it; this is a hard limit and requires rebuilding valgrind if you
/// want to go beyond it. Normally this is not a problem, but in some tests, we
/// produce a lot of threads casually.  Making lots of threads alone might not
/// be a problem _either_, except on OSX, the segments produced for new threads
/// _take a while_ to get reclaimed by the OS. Combined with the fact that libuv
/// schedulers fork off a separate thread for polling fsevents on OSX, we get a
/// perfect storm of creating "too many mappings" for valgrind to handle when
/// running certain stress tests in the runtime.
pub fn limit_thread_creation_due_to_osx_and_valgrind() -> bool {
    (cfg!(target_os="macos")) && running_on_valgrind()
}

pub fn min_stack() -> uint {
    static mut MIN: atomics::AtomicUint = atomics::INIT_ATOMIC_UINT;
    match unsafe { MIN.load(atomics::SeqCst) } {
        0 => {}
        n => return n - 1,
    }
    let amt = os::getenv("RUST_MIN_STACK").and_then(|s| from_str(s.as_slice()));
    let amt = amt.unwrap_or(2 * 1024 * 1024);
    // 0 is our sentinel value, so ensure that we'll never see 0 after
    // initialization has run
    unsafe { MIN.store(amt + 1, atomics::SeqCst); }
    return amt;
}

/// Get's the number of scheduler threads requested by the environment
/// either `RUST_THREADS` or `num_cpus`.
pub fn default_sched_threads() -> uint {
    match os::getenv("RUST_THREADS") {
        Some(nstr) => {
            let opt_n: Option<uint> = FromStr::from_str(nstr.as_slice());
            match opt_n {
                Some(n) if n > 0 => n,
                _ => fail!("`RUST_THREADS` is `{}`, should be a positive integer", nstr)
            }
        }
        None => {
            if limit_thread_creation_due_to_osx_and_valgrind() {
                1
            } else {
                num_cpus()
            }
        }
    }
}
