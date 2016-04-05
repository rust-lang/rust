// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// compile-flags: -Z verbose

// TODO nikomatsakis: test with both verbose and without

trait Foo<'b, 'c, S=u32> {
    fn bar<'a, T>() where T: 'a {}
    fn baz() {}
}

impl<'a,'b,T,S> Foo<'a, 'b, S> for T {}

fn main() {}

fn foo<'z>() where &'z (): Sized {
    let x: () = <i8 as Foo<'static, 'static,  u8>>::bar::<'static, char>;
    //~^ ERROR mismatched types
    //~| expected `()`
    //~| found `fn() {<i8 as Foo<ReStatic, ReStatic, u8>>::bar::<ReStatic, char>}`

    let x: () = <i8 as Foo<'static, 'static,  u32>>::bar::<'static, char>;
    //~^ ERROR mismatched types
    //~| expected `()`
    //~| found `fn() {<i8 as Foo<ReStatic, ReStatic, u32>>::bar::<ReStatic, char>}`

    let x: () = <i8 as Foo<'static, 'static,  u8>>::baz;
    //~^ ERROR mismatched types
    //~| expected `()`
    //~| found `fn() {<i8 as Foo<ReStatic, ReStatic, u8>>::baz}`

    let x: () = foo::<'static>;
    //~^ ERROR mismatched types
    //~| expected `()`
    //~| found `fn() {foo::<ReStatic>}`

    <str as Foo<u8>>::bar;
    //~^ ERROR `str: std::marker::Sized` is not satisfied
}
