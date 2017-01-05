// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo<A: Repr>(<A as Repr>::Data);

impl<A> Copy for Foo<A> where <A as Repr>::Data: Copy { }
impl<A> Clone for Foo<A> where <A as Repr>::Data: Clone {
    fn clone(&self) -> Self { Foo(self.0.clone()) }
}

trait Repr {
    type Data;
}

impl<A> Repr for A {
    type Data = u32;
}

fn main() {
}
