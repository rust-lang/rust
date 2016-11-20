// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(link_cfg)]

#[link(name = "return1", cfg(foo))]
#[link(name = "return2", cfg(bar))]
extern {
    fn my_function() -> i32;
}

fn main() {
    unsafe {
        let v = my_function();
        if cfg!(foo) {
            assert_eq!(v, 1);
        } else if cfg!(bar) {
            assert_eq!(v, 2);
        } else {
            panic!("unknown");
        }
    }
}
