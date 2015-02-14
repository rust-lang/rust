// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Self;
//~^ ERROR expected identifier, found keyword `Self`

struct Bar<'Self>;
//~^ ERROR invalid lifetime name

pub fn main() {
    let Self = 5;
    //~^ ERROR expected identifier, found keyword `Self`

    match 15 {
        Self => (),
        //~^ ERROR expected identifier, found keyword `Self`
        ref Self => (),
        //~^ ERROR expected identifier, found keyword `Self`
        mut Self => (),
        //~^ ERROR expected identifier, found keyword `Self`
        ref mut Self => (),
        //~^ ERROR expected identifier, found keyword `Self`
        Self!() => (),
        //~^ ERROR expected identifier, found keyword `Self`
        Foo { x: Self } => (),
        //~^ ERROR expected identifier, found keyword `Self`
        Foo { Self } => (),
        //~^ ERROR expected identifier, found keyword `Self`
    }
}

use self::Self as Foo;
//~^ ERROR expected identifier, found keyword `Self`

use std::option::Option as Self;
//~^ ERROR expected identifier, found keyword `Self`

extern crate Self;
//~^ ERROR expected identifier, found keyword `Self`

trait Self {}
//~^ ERROR expected identifier, found keyword `Self`
