// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Beginners write `mod.item` when they should write `mod::item`.
// This tests that we suggest the latter when we encounter the former.

pub mod a {
    pub const I: i32 = 1;

    pub fn f() -> i32 { 2 }

    pub mod b {
        pub const J: i32 = 3;

        pub fn g() -> i32 { 4 }
    }
}

fn h1() -> i32 {
    a.I
        //~^ ERROR E0425
        //~| HELP To reference an item from the `a` module, use `a::I`
}

fn h2() -> i32 {
    a.g()
        //~^ ERROR E0425
        //~| HELP To call a function from the `a` module, use `a::g(..)`
}

fn h3() -> i32 {
    a.b.J
        //~^ ERROR E0425
        //~| HELP To reference an item from the `a` module, use `a::b`
}

fn h4() -> i32 {
    a::b.J
        //~^ ERROR E0425
        //~| HELP To reference an item from the `a::b` module, use `a::b::J`
}

fn h5() -> i32 {
    a.b.f()
        //~^ ERROR E0425
        //~| HELP To reference an item from the `a` module, use `a::b`
}

fn h6() -> i32 {
    a::b.f()
        //~^ ERROR E0425
        //~| HELP To call a function from the `a::b` module, use `a::b::f(..)`
}
