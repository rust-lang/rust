// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast
// aux-build:crateresolve4a-1.rs
// aux-build:crateresolve4a-2.rs
// aux-build:crateresolve4b-1.rs
// aux-build:crateresolve4b-2.rs

#[legacy_exports];

mod a {
    #[legacy_exports];
    extern mod crateresolve4b(vers = "0.1");
    fn f() { assert crateresolve4b::f() == 20; }
}

mod b {
    #[legacy_exports];
    extern mod crateresolve4b(vers = "0.2");
    fn f() { assert crateresolve4b::g() == 10; }
}

fn main() {
    a::f();
    b::f();
}
