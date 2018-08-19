// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that methods from shadowed traits cannot be used

mod foo {
    pub trait T { fn f(&self) {} }
    impl T for () {}
}

mod bar { pub use foo::T; }

fn main() {
    pub use bar::*;
    struct T;
    ().f() //~ ERROR no method
}
