// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const A: usize = { 1; 2 };
//~^ ERROR: blocks in constants are limited to items and tail expressions

const B: usize = { { } 2 };
//~^ ERROR: blocks in constants are limited to items and tail expressions

macro_rules! foo {
    () => (()) //~ ERROR: blocks in constants are limited to items and tail expressions
}
const C: usize = { foo!(); 2 };

const D: usize = { let x = 4; 2 };
//~^ ERROR: blocks in constants are limited to items and tail expressions
//~^^ ERROR: blocks in constants are limited to items and tail expressions

enum Foo {
    Bar = { let x = 1; 3 }
    //~^ ERROR: blocks in constants are limited to items and tail expressions
    //~^^ ERROR: blocks in constants are limited to items and tail expressions
}

type Array = [u32; {  let x = 2; 5 }];
//~^ ERROR: blocks in constants are limited to items and tail expressions
//~^^ ERROR: blocks in constants are limited to items and tail expressions

pub fn main() {
    let _: Array = [0; { let x = 3; 5 }];
    //~^ ERROR: blocks in constants are limited to items and tail expressions
    //~^^ ERROR: blocks in constants are limited to items and tail expressions
}
