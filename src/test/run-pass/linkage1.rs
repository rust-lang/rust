// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows
// ignore-macos
// ignore-emscripten
// aux-build:linkage1.rs

#![feature(linkage)]

extern crate linkage1 as other;

extern {
    #[linkage = "extern_weak"]
    static foo: *const isize;
    #[linkage = "extern_weak"]
    static something_that_should_never_exist: *mut isize;
}

fn main() {
    // It appears that the --as-needed flag to linkers will not pull in a dynamic
    // library unless it satisfies a non weak undefined symbol. The 'other' crate
    // is compiled as a dynamic library where it would only be used for a
    // weak-symbol as part of an executable, so the dynamic library would be
    // discarded. By adding and calling `other::bar`, we get around this problem.
    other::bar();

    assert!(!foo.is_null());
    assert_eq!(unsafe { *foo }, 3);
    assert!(something_that_should_never_exist.is_null());
}
