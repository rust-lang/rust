// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:two_macros.rs

macro_rules! foo { () => {} }
macro_rules! macro_one { () => {} }
#[macro_use(macro_two)] extern crate two_macros;

macro_rules! m1 { () => {
    macro_rules! foo { () => {} } //~ ERROR `foo` is already in scope
    //~^ NOTE macro-expanded `macro_rules!`s may not shadow existing macros

    #[macro_use] //~ ERROR `macro_two` is already in scope
    //~^ NOTE macro-expanded `#[macro_use]`s may not shadow existing macros
    extern crate two_macros as __;
}}
m1!(); //~ NOTE in this expansion
       //~| NOTE in this expansion
       //~| NOTE in this expansion
       //~| NOTE in this expansion

foo!();

macro_rules! m2 { () => {
    macro_rules! foo { () => {} }
    foo!();
}}
m2!();
//^ Since `foo` is not used outside this expansion, it is not a shadowing error.

fn main() {}
