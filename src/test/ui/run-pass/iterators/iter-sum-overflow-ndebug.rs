// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C debug_assertions=no

fn main() {
    assert_eq!([1i32, i32::max_value()].iter().sum::<i32>(),
               1i32.wrapping_add(i32::max_value()));
    assert_eq!([2i32, i32::max_value()].iter().product::<i32>(),
               2i32.wrapping_mul(i32::max_value()));

    assert_eq!([1i32, i32::max_value()].iter().cloned().sum::<i32>(),
               1i32.wrapping_add(i32::max_value()));
    assert_eq!([2i32, i32::max_value()].iter().cloned().product::<i32>(),
               2i32.wrapping_mul(i32::max_value()));
}
