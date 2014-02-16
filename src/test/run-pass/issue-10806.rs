// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

pub fn foo() -> int {
    3
}
pub fn bar() -> int {
    4
}

pub mod baz {
    use {foo, bar};
    pub fn quux() -> int {
        foo() + bar()
    }
}

pub mod grault {
    use {foo};
    pub fn garply() -> int {
        foo()
    }
}

pub mod waldo {
    use {};
    pub fn plugh() -> int {
        0
    }
}

pub fn main() {
    let _x = baz::quux();
    let _y = grault::garply();
    let _z = waldo::plugh();
}
