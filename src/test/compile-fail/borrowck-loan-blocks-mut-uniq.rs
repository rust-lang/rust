// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn borrow(v: &int, f: |x: &int|) {
    f(v);
}

fn box_imm() {
    let mut v = box 3;
    let v_ptr = &mut v;
    borrow(*v_ptr,
           |w| { //~ ERROR closure requires unique access to `*v_ptr`
            *v_ptr = box 4; //~ ERROR cannot move `v_ptr`
            assert_eq!(**v_ptr, 3);
            assert_eq!(*w, 4);
        })
}

fn main() {
}
