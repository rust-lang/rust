// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the requirement (in `Bar`) that `T::Bar : 'static` does
// not wind up propagating to `T`.

// pretty-expanded FIXME #23616

pub trait Foo {
    type Bar;

    fn foo(&self) -> Self;
}

pub struct Static<T:'static>(T);

struct Bar<T:Foo>
    where T::Bar : 'static
{
    x: Static<Option<T::Bar>>
}

fn main() { }
