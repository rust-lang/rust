// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod m {
    pub enum E {
        Fn(u8),
        Struct {
            s: u8,
        },
        Unit,
    }

    pub mod n {
        pub(in m) enum Z {
            Fn(u8),
            Struct {
                s: u8,
            },
            Unit,
        }
    }

    use m::n::Z; // OK, only the type is imported

    fn f() {
        n::Z;
        //~^ ERROR expected value, found enum `n::Z`
        Z;
        //~^ ERROR expected value, found enum `Z`
        let _: Z = Z::Fn;
        //~^ ERROR mismatched types
        let _: Z = Z::Struct;
        //~^ ERROR expected value, found struct variant `Z::Struct`
        let _ = Z::Unit();
        //~^ ERROR expected function, found enum variant `Z::Unit`
        let _ = Z::Unit {};
        // This is ok, it is equivalent to not having braces
    }
}

use m::E; // OK, only the type is imported

fn main() {
    let _: E = m::E;
    //~^ ERROR expected value, found enum `m::E`
    let _: E = m::E::Fn;
    //~^ ERROR mismatched types
    let _: E = m::E::Struct;
    //~^ ERROR expected value, found struct variant `m::E::Struct`
    let _: E = m::E::Unit();
    //~^ ERROR expected function, found enum variant `m::E::Unit`
    let _: E = E;
    //~^ ERROR expected value, found enum `E`
    let _: E = E::Fn;
    //~^ ERROR mismatched types
    let _: E = E::Struct;
    //~^ ERROR expected value, found struct variant `E::Struct`
    let _: E = E::Unit();
    //~^ ERROR expected function, found enum variant `E::Unit`
    let _: Z = m::n::Z;
    //~^ ERROR cannot find type `Z` in this scope
    //~| ERROR expected value, found enum `m::n::Z`
    //~| ERROR enum `Z` is private
    let _: Z = m::n::Z::Fn;
    //~^ ERROR cannot find type `Z` in this scope
    //~| ERROR enum `Z` is private
    let _: Z = m::n::Z::Struct;
    //~^ ERROR cannot find type `Z` in this scope
    //~| ERROR expected value, found struct variant `m::n::Z::Struct`
    //~| ERROR enum `Z` is private
    let _: Z = m::n::Z::Unit {};
    //~^ ERROR cannot find type `Z` in this scope
    //~| ERROR enum `Z` is private
}
