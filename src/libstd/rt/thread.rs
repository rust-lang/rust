// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use ops::Drop;

#[allow(non_camel_case_types)] // runtime type
type raw_thread = libc::c_void;

pub struct Thread {
    main: ~fn(),
    raw_thread: *raw_thread,
    joined: bool
}

impl Thread {
    pub fn start(main: ~fn()) -> Thread {
        fn substart(main: &~fn()) -> *raw_thread {
            #[fixed_stack_segment]; #[inline(never)];

            unsafe { rust_raw_thread_start(main) }
        }
        let raw = substart(&main);
        Thread {
            main: main,
            raw_thread: raw,
            joined: false
        }
    }

    pub fn join(self) {
        #[fixed_stack_segment]; #[inline(never)];

        assert!(!self.joined);
        let mut this = self;
        unsafe { rust_raw_thread_join(this.raw_thread); }
        this.joined = true;
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
    pub fn rust_raw_thread_start(f: &(~fn())) -> *raw_thread;
    pub fn rust_raw_thread_join(thread: *raw_thread);
    pub fn rust_raw_thread_delete(thread: *raw_thread);
}
