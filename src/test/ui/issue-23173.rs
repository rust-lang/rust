// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Token { LeftParen, RightParen, Plus, Minus, /* etc */ }
//~^ NOTE variant `Homura` not found here
struct Struct {
    //~^ NOTE function or associated item `method` not found for this
    //~| NOTE function or associated item `method` not found for this
    //~| NOTE associated item `Assoc` not found for this
    a: usize,
}

fn use_token(token: &Token) { unimplemented!() }

fn main() {
    use_token(&Token::Homura);
    //~^ ERROR no variant named `Homura`
    //~| NOTE variant not found in `Token`
    Struct::method();
    //~^ ERROR no function or associated item named `method` found for type
    //~| NOTE function or associated item not found in `Struct`
    Struct::method;
    //~^ ERROR no function or associated item named `method` found for type
    //~| NOTE function or associated item not found in `Struct`
    Struct::Assoc;
    //~^ ERROR no associated item named `Assoc` found for type `Struct` in
    //~| NOTE associated item not found in `Struct`
}
