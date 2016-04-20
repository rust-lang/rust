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
// revisions: verbose normal
//
//[verbose] compile-flags: -Z verbose

trait Foo<'b, 'c, S=u32> {
    fn bar<'a, T>() where T: 'a {}
    fn baz() {}
}

impl<'a,'b,T,S> Foo<'a, 'b, S> for T {}

fn main() {}

fn foo<'z>() where &'z (): Sized {
    let x: () = <i8 as Foo<'static, 'static,  u8>>::bar::<'static, char>;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| expected type `()`
    //[verbose]~| found type `fn() {<i8 as Foo<ReStatic, ReStatic, u8>>::bar::<ReStatic, char>}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| expected type `()`
    //[normal]~| found type `fn() {<i8 as Foo<'static, 'static, u8>>::bar::<'static, char>}`


    let x: () = <i8 as Foo<'static, 'static,  u32>>::bar::<'static, char>;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| expected type `()`
    //[verbose]~| found type `fn() {<i8 as Foo<ReStatic, ReStatic, u32>>::bar::<ReStatic, char>}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| expected type `()`
    //[normal]~| found type `fn() {<i8 as Foo<'static, 'static>>::bar::<'static, char>}`

    let x: () = <i8 as Foo<'static, 'static,  u8>>::baz;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| expected type `()`
    //[verbose]~| found type `fn() {<i8 as Foo<ReStatic, ReStatic, u8>>::baz}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| expected type `()`
    //[normal]~| found type `fn() {<i8 as Foo<'static, 'static, u8>>::baz}`

    let x: () = foo::<'static>;
    //[verbose]~^ ERROR mismatched types
    //[verbose]~| expected type `()`
    //[verbose]~| found type `fn() {foo::<ReStatic>}`
    //[normal]~^^^^ ERROR mismatched types
    //[normal]~| expected type `()`
    //[normal]~| found type `fn() {foo::<'static>}`

    <str as Foo<u8>>::bar;
    //[verbose]~^ ERROR `str: std::marker::Sized` is not satisfied
    //[normal]~^^ ERROR `str: std::marker::Sized` is not satisfied
}
