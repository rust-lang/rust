// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(transmute_from_fn_item_types)]

use std::mem;

fn main() {
    unsafe {
        let u = mem::transmute(main);
        let p = mem::transmute(main);
        let f = mem::transmute(main);
        let tuple: (usize, *mut (), fn()) = (u, p, f);
        assert_eq!(mem::transmute::<_, [usize; 3]>(tuple), [main as usize; 3]);

        mem::transmute::<_, usize>(main);
        mem::transmute::<_, *mut ()>(main);
        mem::transmute::<_, fn()>(main);
    }
}
