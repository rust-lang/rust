// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that method bounds declared on traits/impls in a cross-crate
// scenario work. This is the libary portion of the test.

pub enum MaybeOwned<'a> {
    Owned(int),
    Borrowed(&'a int)
}

struct Inv<'a> { // invariant w/r/t 'a
    x: &'a mut &'a int
}

// I encountered a bug at some point with encoding the IntoMaybeOwned
// trait, so I'll use that as the template for this test.
pub trait IntoMaybeOwned<'a> {
    fn into_maybe_owned(self) -> MaybeOwned<'a>;
    fn bigger_region<'b:'a>(self, b: Inv<'b>);
}

impl<'a> IntoMaybeOwned<'a> for Inv<'a> {
    fn into_maybe_owned(self) -> MaybeOwned<'a> { fail!() }
    fn bigger_region<'b:'a>(self, b: Inv<'b>) { fail!() }
}
