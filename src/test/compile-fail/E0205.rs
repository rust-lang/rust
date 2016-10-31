// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo {
    Bar(Vec<u32>),
    //~^ NOTE this variant doesn't implement `Copy`
    Baz,
}

impl Copy for Foo { }
//~^ ERROR the trait `Copy` may not be implemented for type `Foo`
//~| NOTE variant `Foo::Bar` doesn't implement `Copy`

#[derive(Copy)]
//~^ ERROR the trait `Copy` may not be implemented for type `Foo2<'a>`
//~| NOTE variant `Foo2::Bar` doesn't implement `Copy`
//~| NOTE in this expansion of #[derive(Copy)]
enum Foo2<'a> {
    Bar(&'a mut bool),
    //~^ NOTE this variant doesn't implement `Copy`
    Baz,
}

fn main() {
}
