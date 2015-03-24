// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Given `<expr> as Box<Trait>`, we should be able to infer that a
// `Box<_>` is the expected type.

// pretty-expanded FIXME #23616

trait Foo { fn foo(&self) -> u32; }
impl Foo for u32 { fn foo(&self) -> u32 { *self } }

// (another impl to ensure trait-matching cannot just choose from a singleton set)
impl Foo for  () { fn foo(&self) -> u32 { -176 } }

trait Boxed { fn make() -> Self; }
impl Boxed for Box<u32> { fn make() -> Self { Box::new(7) } }

// (another impl to ensure trait-matching cannot just choose from a singleton set)
impl Boxed for () { fn make() -> Self { () } }

fn boxed_foo() {
    let b7 = Boxed::make() as Box<Foo>;
    assert_eq!(b7.foo(), 7);
}

trait Refed<'a,T> { fn make(&'a T) -> Self; }
impl<'a> Refed<'a, u32> for &'a u32 { fn make(x: &'a u32) -> Self { x } }

// (another impl to ensure trait-matching cannot just choose from a singleton set)
impl<'a,'b> Refed<'a, ()> for &'b () { fn make(_: &'a ()) -> Self { static U: () = (); &U } }

fn refed_foo() {
    let a = 8;
    let b7 = Refed::make(&a) as &Foo;
    assert_eq!(b7.foo(), 8);
}

fn check_subtyping_works() {
    fn inner<'short, 'long:'short>(_s: &'short u32,
                                   l: &'long u32) -> &'short (Foo+'short) {
        Refed::make(l) as &Foo
    }

    let a = 9;
    let b = 10;
    let r = inner(&b, &a);
    assert_eq!(r.foo(), 9);
}

pub fn main() {
    boxed_foo();
    refed_foo();
    check_subtyping_works();
}
