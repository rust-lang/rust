// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:proc_macro_def.rs
// ignore-stage1
// ignore-cross-compile

#![feature(plugin, custom_attribute)]
#![feature(type_macros)]

#![plugin(proc_macro_def)]

#[attr_tru]
fn f1() -> bool {
    return false;
}

#[attr_identity]
fn f2() -> bool {
    return identity!(true);
}

fn f3() -> identity!(bool) {
    ret_tru!();
}

fn f4(x: bool) -> bool {
    match x {
        identity!(true) => false,
        identity!(false) => true,
    }
}

fn main() {
    assert!(f1());
    assert!(f2());
    assert!(tru!());
    assert!(f3());
    assert!(identity!(5 == 5));
    assert!(f4(false));
}
