// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Currently, we permit you to assign to individual fields of a mut
// var, but we do not permit you to use the complete var afterwards.
// We hope to fix this at some point.
//
// FIXME(#54987)

fn assign_both_fields_and_use() {
    let mut x: (u32, u32);
    x.0 = 1;
    x.1 = 22;
    drop(x.0); //~ ERROR
    drop(x.1); //~ ERROR
}

fn assign_both_fields_the_use_var() {
    let mut x: (u32, u32);
    x.0 = 1;
    x.1 = 22;
    drop(x); //~ ERROR
}

fn main() { }
