// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that projections don't count as constraining type parameters.

struct S<T>(T);

trait Tr { type Assoc; fn test(); }

impl<T: Tr> S<T::Assoc> {
//~^ ERROR the type parameter `T` is not constrained
    fn foo(self, _: T) {
        T::test();
    }
}

trait Trait1<T> { type Bar; }
trait Trait2<'x> { type Foo; }

impl<'a,T: Trait2<'a>> Trait1<<T as Trait2<'a>>::Foo> for T {
//~^ ERROR the lifetime parameter `'a` is not constrained
    type Bar = &'a ();
}

fn main() {}
