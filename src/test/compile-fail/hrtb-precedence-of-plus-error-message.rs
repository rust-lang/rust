// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

// Test that we suggest the correct parentheses

trait Bar {
    fn dummy(&self) { }
}

struct Foo<'a> {
    a: &'a Bar+'a,
        //~^ ERROR E0171
        //~^^ NOTE perhaps you meant `&'a (Bar + 'a)`?

    b: &'a mut Bar+'a,
        //~^ ERROR E0171
        //~^^ NOTE perhaps you meant `&'a mut (Bar + 'a)`?

    c: Box<Bar+'a>, // OK, no paren needed in this context

    d: fn() -> Bar+'a,
        //~^ ERROR E0171
        //~^^ NOTE perhaps you forgot parentheses
        //~^^^ WARN deprecated syntax
}

fn main() { }
