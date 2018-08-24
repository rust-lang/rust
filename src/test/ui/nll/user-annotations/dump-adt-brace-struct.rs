// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Unit test for the "user substitutions" that are annotated on each
// node.

// compile-flags:-Zverbose

#![allow(warnings)]
#![feature(nll)]
#![feature(rustc_attrs)]

struct SomeStruct<T> { t: T }

#[rustc_dump_user_substs]
fn main() {
    SomeStruct { t: 22 }; // Nothing given, no annotation.

    SomeStruct::<_> { t: 22 }; // Nothing interesting given, no annotation.

    SomeStruct::<u32> { t: 22 }; //~ ERROR [u32]
}
