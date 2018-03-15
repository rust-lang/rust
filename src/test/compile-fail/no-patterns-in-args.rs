// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern {
    fn f1(mut arg: u8); //~ ERROR patterns aren't allowed in foreign function declarations
                        //~^ NOTE pattern not allowed in foreign function
                        //~| NOTE this is a recent error
    fn f2(&arg: u8); //~ ERROR patterns aren't allowed in foreign function declarations
                     //~^ NOTE pattern not allowed in foreign function
    fn f3(arg @ _: u8); //~ ERROR patterns aren't allowed in foreign function declarations
                        //~^ NOTE pattern not allowed in foreign function
                        //~| NOTE this is a recent error
    fn g1(arg: u8); // OK
    fn g2(_: u8); // OK
    // fn g3(u8); // Not yet
}

type A1 = fn(mut arg: u8); //~ ERROR patterns aren't allowed in function pointer types
                           //~^ NOTE this is a recent error
type A2 = fn(&arg: u8); //~ ERROR patterns aren't allowed in function pointer types
                        //~^ NOTE this is a recent error
type B1 = fn(arg: u8); // OK
type B2 = fn(_: u8); // OK
type B3 = fn(u8); // OK

fn main() {}
