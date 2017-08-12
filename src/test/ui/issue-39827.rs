// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(core_intrinsics)]

use std::intrinsics::{ volatile_copy_memory, volatile_store, volatile_load,
                       volatile_copy_nonoverlapping_memory,
                       volatile_set_memory };

fn main () {
    let mut dst_pair = (1, 2);
    let src_pair = (3, 4);
    let mut dst_empty = ();
    let src_empty = ();

    const COUNT_0: usize = 0;
    const COUNT_100: usize = 100;

    unsafe {
        volatile_copy_memory(&mut dst_pair, &dst_pair, COUNT_0);
        volatile_copy_nonoverlapping_memory(&mut dst_pair, &src_pair, 0);
        volatile_copy_memory(&mut dst_empty, &dst_empty, 100);
        volatile_copy_nonoverlapping_memory(&mut dst_empty, &src_empty,
                                            COUNT_100);
        volatile_set_memory(&mut dst_empty, 0, COUNT_100);
        volatile_set_memory(&mut dst_pair, 0, COUNT_0);
        volatile_store(&mut dst_empty, ());
        volatile_store(&mut dst_empty, src_empty);
        volatile_load(&src_empty);
    }
}
