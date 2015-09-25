// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that static methods don't get to assume `Self` is well-formed

trait Foo<'a, 'b>: Sized {
    fn make_me() -> Self { loop {} }
    fn static_evil(u: &'a u32) -> &'b u32;
}

struct Evil<'a, 'b: 'a>(Option<&'a &'b ()>);

impl<'a, 'b> Foo<'a, 'b> for Evil<'a, 'b> {
    fn make_me() -> Self { Evil(None) }
    fn static_evil(u: &'a u32) -> &'b u32 {
        u //~ ERROR cannot infer an appropriate lifetime
    }
}

struct IndirectEvil<'a, 'b: 'a>(Option<&'a &'b ()>);

impl<'a, 'b> Foo<'a, 'b> for IndirectEvil<'a, 'b> {
    fn make_me() -> Self { IndirectEvil(None) }
    fn static_evil(u: &'a u32) -> &'b u32 {
        let me = Self::make_me(); //~ ERROR lifetime bound not satisfied
        loop {} // (`me` could be used for the lifetime transmute).
    }
}

fn main() {}
