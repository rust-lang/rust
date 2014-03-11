// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-win32
// ignore-fast
// ignore-android
// ignore-macos
// aux-build:linkage1.rs

#[feature(linkage)];

extern crate other = "linkage1";

extern {
    #[linkage = "extern_weak"]
    static foo: *int;
    #[linkage = "extern_weak"]
    static something_that_should_never_exist: *mut int;
}

fn main() {
    assert!(!foo.is_null());
    assert_eq!(unsafe { *foo }, 3);
    assert!(something_that_should_never_exist.is_null());
}
