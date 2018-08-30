// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that coercions unify the expected return type of a polymorphic
// function call, instead of leaving the type variables as they were.

// pretty-expanded FIXME #23616

struct Foo;
impl Foo {
    fn foo<T>(self, x: T) -> Option<T> { Some(x) }
}

pub fn main() {
    let _: Option<fn()> = Some(main);
    let _: Option<fn()> = Foo.foo(main);

    // The same two cases, with implicit type variables made explicit.
    let _: Option<fn()> = Some::<_>(main);
    let _: Option<fn()> = Foo.foo::<_>(main);
}
