// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//compile-flags: -Z emit-end-regions -Z borrowck-mir -Z mir

#![allow(unused_assignments)]

struct Wrap<'a> { w: &'a mut u32 }

fn foo() {
    let mut x = 22u64;
    let wrapper = Wrap { w: &mut x };
    x += 1;  //~ ERROR cannot assign to `x`
    *wrapper.w += 1;
}

fn main() { }