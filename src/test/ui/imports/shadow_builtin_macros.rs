// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:two_macros.rs

#![feature(use_extern_macros)]

mod foo {
    extern crate two_macros;
    pub use self::two_macros::m as panic;
}

mod m1 {
    use foo::panic; // ok
    fn f() { panic!(); }
}

mod m2 {
    use foo::*;
    fn f() { panic!(); } //~ ERROR ambiguous
}

mod m3 {
    ::two_macros::m!(use foo::panic;);
    fn f() { panic!(); } //~ ERROR ambiguous
}

mod m4 {
    macro_rules! panic { () => {} } // ok
    panic!();
}

mod m5 {
    macro_rules! m { () => {
        macro_rules! panic { () => {} } //~ ERROR `panic` is already in scope
    } }
    m!();
    panic!();
}

#[macro_use(n)]
extern crate two_macros;
mod bar {
    pub use two_macros::m as n;
}

mod m6 {
    use bar::n; // ok
    n!();
}

mod m7 {
    use bar::*;
    n!(); //~ ERROR ambiguous
}

fn main() {}
