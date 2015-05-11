// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that when there are vacuous predicates in the environment
// (which make a fn uncallable) we don't erroneously cache those and
// then consider them satisfied elsewhere. The current technique for
// doing this is just to filter "global" predicates out of the
// environment, which means that we wind up with an error in the
// function `vacuous`, because even though `i32: Bar<u32>` is implied
// by its where clause, that where clause never holds.

trait Foo<X,Y>: Bar<X> {
}

trait Bar<X> { }

fn vacuous<A>()
    where i32: Foo<u32, A>
{
    // vacuous could never be called, because it requires that i32:
    // Bar<u32>. But the code doesn't check that this could never be
    // satisfied.
    require::<i32, u32>();
    //~^ ERROR the trait `Bar<u32>` is not implemented for the type `i32`
}

fn require<A,B>()
    where A: Bar<B>
{
}

fn main() {
    require::<i32, u32>();
}
