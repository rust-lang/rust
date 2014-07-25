// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that the trait matching code takes lifetime parameters into account.
// (Issue #15517.)

struct Foo<'a,'b> {
    x: &'a int,
    y: &'b int,
}

trait Tr {
    fn foo(x: Self) {}
}

impl<'a,'b> Tr for Foo<'a,'b> {
    fn foo(x: Foo<'b,'a>) {
        //~^ ERROR method not compatible with trait
        //~^^ ERROR method not compatible with trait
    }
}

fn main(){}
