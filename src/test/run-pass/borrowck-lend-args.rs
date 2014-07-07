// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn borrow(_v: &int) {}

fn borrow_from_arg_imm_ref(v: Box<int>) {
    borrow(&*v);
}

fn borrow_from_arg_mut_ref(v: &mut Box<int>) {
    borrow(&**v);
}

fn borrow_from_arg_copy(v: Box<int>) {
    borrow(&*v);
}

pub fn main() {
}
