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
    use foo::*; //~ NOTE `panic` could refer to the name imported here
    fn f() { panic!(); } //~ ERROR ambiguous
    //~| NOTE `panic` is also a builtin macro
    //~| NOTE consider adding an explicit import of `panic` to disambiguate
}

mod m3 {
    ::two_macros::m!(use foo::panic;); //~ NOTE `panic` could refer to the name imported here
    fn f() { panic!(); } //~ ERROR ambiguous
    //~| NOTE `panic` is also a builtin macro
    //~| NOTE macro-expanded macro imports do not shadow
}

mod m4 {
    macro_rules! panic { () => {} } // ok
    panic!();
}

mod m5 {
    macro_rules! m { () => {
        macro_rules! panic { () => {} } //~ ERROR `panic` is already in scope
        //~| NOTE macro-expanded `macro_rules!`s may not shadow existing macros
    } }
    m!(); //~ NOTE in this expansion
    //~| NOTE in this expansion
    panic!();
}

#[macro_use(n)] //~ NOTE `n` could also refer to the name imported here
extern crate two_macros;
mod bar {
    pub use two_macros::m as n;
}

mod m6 {
    use bar::n; // ok
    n!();
}

mod m7 {
    use bar::*; //~ NOTE `n` could refer to the name imported here
    n!(); //~ ERROR ambiguous
    //~| NOTE consider adding an explicit import of `n` to disambiguate
}

fn main() {}
