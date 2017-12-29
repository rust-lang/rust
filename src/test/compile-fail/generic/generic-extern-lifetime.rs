// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test to make sure the names of the lifetimes are correctly resolved
// in extern blocks.

extern {
   pub fn life<'a>(x:&'a i32);
   pub fn life2<'b>(x:&'a i32, y:&'b i32); //~ ERROR use of undeclared lifetime name `'a`
   pub fn life3<'a>(x:&'a i32, y:&i32) -> &'a i32;
   pub fn life4<'b>(x: for<'c> fn(&'a i32)); //~ ERROR use of undeclared lifetime name `'a`
   pub fn life5<'b>(x: for<'c> fn(&'b i32));
   pub fn life6<'b>(x: for<'c> fn(&'c i32));
   pub fn life7<'b>() -> for<'c> fn(&'a i32); //~ ERROR use of undeclared lifetime name `'a`
   pub fn life8<'b>() -> for<'c> fn(&'b i32);
   pub fn life9<'b>() -> for<'c> fn(&'c i32);
}
fn main() {}
