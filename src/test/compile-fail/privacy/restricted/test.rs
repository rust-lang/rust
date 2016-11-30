// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:pub_restricted.rs

#![feature(pub_restricted)]
#![deny(private_in_public)]
#![allow(warnings)]
extern crate pub_restricted;

mod foo {
    pub mod bar {
        pub(super) fn f() {}
        #[derive(Default)]
        pub struct S {
            pub(super) x: i32,
        }
        impl S {
            pub(super) fn f(&self) {}
            pub(super) fn g() {}
        }
    }
    fn f() {
        use foo::bar::S;
        pub(self) use foo::bar::f; // ok
        pub(super) use foo::bar::f as g; //~ ERROR cannot be reexported
        S::default().x; // ok
        S::default().f(); // ok
        S::g(); // ok
    }
}

fn f() {
    use foo::bar::S;
    use foo::bar::f; //~ ERROR private
    S::default().x; //~ ERROR private
    S::default().f(); //~ ERROR private
    S::g(); //~ ERROR private
}

fn main() {
    use pub_restricted::Universe;
    use pub_restricted::Crate; //~ ERROR private

    let u = Universe::default();
    let _ = u.x;
    let _ = u.y; //~ ERROR private
    u.f();
    u.g(); //~ ERROR private
}

mod pathological {
    pub(bad::path) mod m1 {} //~ ERROR failed to resolve. Maybe a missing `extern crate bad;`?
    pub(foo) mod m2 {} //~ ERROR visibilities can only be restricted to ancestor modules
}
