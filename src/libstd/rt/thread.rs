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
use libc;
use ops::Drop;
use unstable::raw;
use uint;

#[allow(non_camel_case_types)] // runtime type
type raw_thread = libc::c_void;

pub struct Thread {
    priv main: ~fn(),
    priv raw_thread: *raw_thread,
    priv joined: bool
}

impl Thread {
    #[fixed_stack_segment] #[inline(never)]
    pub fn start(main: ~fn()) -> Thread {
        // This is the starting point of rust os threads. The first thing we do
        // is make sure that we don't trigger __morestack (also why this has a
        // no_split_stack annotation), and then we re-build the main function
        // and invoke it from there.
        #[no_split_stack]
        extern "C" fn thread_start(code: *(), env: *()) {
            use rt::context;
            unsafe {
                context::record_stack_bounds(0, uint::max_value);
                let f: &fn() = cast::transmute(raw::Closure {
                    code: code,
                    env: env,
                });
                f();
            }
        }

        let raw_thread = unsafe {
            let c: raw::Closure = cast::transmute_copy(&main);
            let raw::Closure { code, env } = c;
            rust_raw_thread_start(thread_start, code, env)
        };
        Thread {
            main: main,
            raw_thread: raw_thread,
            joined: false,
        }
    }

    pub fn join(mut self) {
        #[fixed_stack_segment]; #[inline(never)];

        assert!(!self.joined);
        unsafe { rust_raw_thread_join(self.raw_thread); }
        self.joined = true;
    }
}

impl Drop for Thread {
    fn drop(&mut self) {
        #[fixed_stack_segment]; #[inline(never)];

        assert!(self.joined);
        unsafe { rust_raw_thread_delete(self.raw_thread) }
    }
}

extern {
    fn rust_raw_thread_start(f: extern "C" fn(*(), *()),
                             code: *(), env: *()) -> *raw_thread;
    fn rust_raw_thread_join(thread: *raw_thread);
    fn rust_raw_thread_delete(thread: *raw_thread);
}
