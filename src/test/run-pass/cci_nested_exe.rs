// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast - check-fast doesn't understand aux-build
// aux-build:cci_nested_lib.rs

extern mod cci_nested_lib;
use cci_nested_lib::*;

pub fn main() {
    let lst = new_int_alist();
    alist_add(lst, 22, ~"hi");
    alist_add(lst, 44, ~"ho");
    assert!(alist_get(lst, 22) == ~"hi");
    assert!(alist_get(lst, 44) == ~"ho");

    let lst = new_int_alist_2();
    alist_add(lst, 22, ~"hi");
    alist_add(lst, 44, ~"ho");
    assert!(alist_get(lst, 22) == ~"hi");
    assert!(alist_get(lst, 44) == ~"ho");
}
