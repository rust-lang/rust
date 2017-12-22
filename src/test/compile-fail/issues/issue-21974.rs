// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that (for now) we report an ambiguity error here, because
// specific trait relationships are ignored for the purposes of trait
// matching. This behavior should likely be improved such that this
// test passes. See #21974 for more details.

trait Foo {
    fn foo(self);
}

fn foo<'a,'b,T>(x: &'a T, y: &'b T) //~ ERROR type annotations required
    where &'a T : Foo,
          &'b T : Foo
{
    x.foo();
    y.foo();
}

fn main() { }
