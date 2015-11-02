// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use mem;
use sys::rt;
use thread::{self, Thread};

#[cfg(not(test))]
#[lang = "start"]
fn lang_start(main: *const u8, argc: isize, argv: *const *const u8) -> isize {
    let failed = unsafe {
        rt::run_main(|| {
            thread::info::set_current_thread(Thread::new(Some("<main>".into())));
            let res = thread::catch_panic(mem::transmute::<_, fn()>(main));
            rt::std_cleanup();
            res
        }, argc, argv).is_err()
    };

    if failed {
        101
    } else {
        0
    }
}
