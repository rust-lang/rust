// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C codegen-units=3
// aux-build:sepcomp-extern-lib.rs

// Test accessing external items from multiple compilation units.

#[link(name = "sepcomp-extern-lib")]
extern {
    #[allow(ctypes)]
    fn foo() -> uint;
}

fn call1() -> uint {
    unsafe { foo() }
}

mod a {
    pub fn call2() -> uint {
        unsafe { ::foo() }
    }
}

mod b {
    pub fn call3() -> uint {
        unsafe { ::foo() }
    }
}

fn main() {
    assert_eq!(call1(), 1234);
    assert_eq!(a::call2(), 1234);
    assert_eq!(b::call3(), 1234);
}
