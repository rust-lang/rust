// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only

struct S;

impl S {
    fn foo(&self, &str bar) {}
    //~^ ERROR expected one of `:` or `@`
    //~| HELP declare the type after the parameter binding
    //~| SUGGESTION <identifier>: <type>
}

fn baz(S quux, xyzzy: i32) {}
//~^ ERROR expected one of `:` or `@`
//~| HELP declare the type after the parameter binding
//~| SUGGESTION <identifier>: <type>

fn one(i32 a b) {}
//~^ ERROR expected one of `:` or `@`

fn pattern((i32, i32) (a, b)) {}
//~^ ERROR expected `:`

fn fizz(i32) {}
//~^ ERROR expected one of `:` or `@`

fn missing_colon(quux S) {}
//~^ ERROR expected one of `:` or `@`
//~| HELP declare the type after the parameter binding
//~| SUGGESTION <identifier>: <type>

fn main() {}
