// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn outer<T>() {
    #[allow(dead_code)]
    fn inner() -> u32 {
        8675309
    }
}

extern "C" fn outer_foreign<T>() {
    #[allow(dead_code)]
    fn inner() -> u32 {
        11235813
    }
}

fn main() {
    outer::<isize>();
    outer::<usize>();
    outer_foreign::<isize>();
    outer_foreign::<usize>();
}
