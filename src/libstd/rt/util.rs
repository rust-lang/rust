// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15677

use prelude::v1::*;

use cmp;
use fmt;
use intrinsics;
use libc::{self, uintptr_t};
use os;
use slice;
use str;
use sync::atomic::{self, Ordering};

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
    static MIN: atomic::AtomicUsize = atomic::ATOMIC_USIZE_INIT;
    match MIN.load(Ordering::SeqCst) {
        0 => {}
        n => return n - 1,
    }
    let amt = os::getenv("RUST_MIN_STACK").and_then(|s| s.parse());
    let amt = amt.unwrap_or(2 * 1024 * 1024);
    // 0 is our sentinel value, so ensure that we'll never see 0 after
    // initialization has run
    MIN.store(amt + 1, Ordering::SeqCst);
    return amt;
}

/// Get's the number of scheduler threads requested by the environment
/// either `RUST_THREADS` or `num_cpus`.
pub fn default_sched_threads() -> uint {
    match os::getenv("RUST_THREADS") {
        Some(nstr) => {
            let opt_n: Option<uint> = nstr.parse();
            match opt_n {
                Some(n) if n > 0 => n,
                _ => panic!("`RUST_THREADS` is `{}`, should be a positive integer", nstr)
            }
        }
        None => {
            if limit_thread_creation_due_to_osx_and_valgrind() {
                1
            } else {
                os::num_cpus()
            }
        }
    }
}

// Indicates whether we should perform expensive sanity checks, including rtassert!
//
// FIXME: Once the runtime matures remove the `true` below to turn off rtassert,
//        etc.
pub const ENFORCE_SANITY: bool = true || !cfg!(rtopt) || cfg!(rtdebug) ||
                                  cfg!(rtassert);

#[allow(missing_copy_implementations)]
pub struct Stdio(libc::c_int);

#[allow(non_upper_case_globals)]
pub const Stdout: Stdio = Stdio(libc::STDOUT_FILENO);
#[allow(non_upper_case_globals)]
pub const Stderr: Stdio = Stdio(libc::STDERR_FILENO);

impl Stdio {
    pub fn write_bytes(&mut self, data: &[u8]) {
        #[cfg(unix)]
        type WriteLen = libc::size_t;
        #[cfg(windows)]
        type WriteLen = libc::c_uint;
        unsafe {
            let Stdio(fd) = *self;
            libc::write(fd,
                        data.as_ptr() as *const libc::c_void,
                        data.len() as WriteLen);
        }
    }
}

impl fmt::Writer for Stdio {
    fn write_str(&mut self, data: &str) -> fmt::Result {
        self.write_bytes(data.as_bytes());
        Ok(()) // yes, we're lying
    }
}

pub fn dumb_print(args: fmt::Arguments) {
    let _ = Stderr.write_fmt(args);
}

pub fn abort(args: fmt::Arguments) -> ! {
    use fmt::Writer;

    struct BufWriter<'a> {
        buf: &'a mut [u8],
        pos: uint,
    }
    impl<'a> fmt::Writer for BufWriter<'a> {
        fn write_str(&mut self, bytes: &str) -> fmt::Result {
            let left = &mut self.buf[self.pos..];
            let to_write = &bytes.as_bytes()[..cmp::min(bytes.len(), left.len())];
            slice::bytes::copy_memory(left, to_write);
            self.pos += to_write.len();
            Ok(())
        }
    }

    // Convert the arguments into a stack-allocated string
    let mut msg = [0u8; 512];
    let mut w = BufWriter { buf: &mut msg, pos: 0 };
    let _ = write!(&mut w, "{}", args);
    let msg = str::from_utf8(&w.buf[..w.pos]).unwrap_or("aborted");
    let msg = if msg.is_empty() {"aborted"} else {msg};
    rterrln!("fatal runtime error: {}", msg);
    unsafe { intrinsics::abort(); }
}

pub unsafe fn report_overflow() {
    use thread::Thread;

    // See the message below for why this is not emitted to the
    // ^ Where did the message below go?
    // task's logger. This has the additional conundrum of the
    // logger may not be initialized just yet, meaning that an FFI
    // call would happen to initialized it (calling out to libuv),
    // and the FFI call needs 2MB of stack when we just ran out.

    rterrln!("\nthread '{}' has overflowed its stack",
             Thread::current().name().unwrap_or("<unknown>"));
}
