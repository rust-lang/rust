// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z continue-parse-after-error

struct Self;
//~^ ERROR expected identifier, found keyword `Self`

struct Bar<'Self>;
//~^ ERROR lifetimes cannot use keyword names

pub fn main() {
    match 15 {
        ref Self => (),
        //~^ ERROR expected identifier, found keyword `Self`
        mut Self => (),
        //~^ ERROR expected identifier, found keyword `Self`
        ref mut Self => (),
        //~^ ERROR expected identifier, found keyword `Self`
        Self!() => (),
        //~^ ERROR macro undefined: 'Self!'
        Foo { Self } => (),
        //~^ ERROR expected identifier, found keyword `Self`
    }
}

mod m1 {
    extern crate core as Self;
    //~^ ERROR expected identifier, found keyword `Self`
}

mod m2 {
    use std::option::Option as Self;
    //~^ ERROR expected identifier, found keyword `Self`
}

mod m3 {
    trait Self {}
    //~^ ERROR expected identifier, found keyword `Self`
}
