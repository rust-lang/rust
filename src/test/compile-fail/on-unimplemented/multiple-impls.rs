// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test if the on_unimplemented message override works

#![feature(on_unimplemented)]
#![feature(rustc_attrs)]

struct Foo<T>(T);
struct Bar<T>(T);

#[rustc_on_unimplemented = "trait message"]
trait Index<Idx: ?Sized> {
    type Output: ?Sized;
    fn index(&self, index: Idx) -> &Self::Output;
}

#[rustc_on_unimplemented = "on impl for Foo"]
impl Index<Foo<usize>> for [i32] {
    type Output = i32;
    fn index(&self, _index: Foo<usize>) -> &i32 {
        loop {}
    }
}

#[rustc_on_unimplemented = "on impl for Bar"]
impl Index<Bar<usize>> for [i32] {
    type Output = i32;
    fn index(&self, _index: Bar<usize>) -> &i32 {
        loop {}
    }
}

#[rustc_error]
fn main() {
    Index::index(&[] as &[i32], 2u32);
    //~^ ERROR E0277
    //~| NOTE the trait `Index<u32>` is not implemented for `[i32]`
    //~| NOTE trait message
    //~| NOTE required by
    Index::index(&[] as &[i32], Foo(2u32));
    //~^ ERROR E0277
    //~| NOTE the trait `Index<Foo<u32>>` is not implemented for `[i32]`
    //~| NOTE on impl for Foo
    //~| NOTE required by
    Index::index(&[] as &[i32], Bar(2u32));
    //~^ ERROR E0277
    //~| NOTE the trait `Index<Bar<u32>>` is not implemented for `[i32]`
    //~| NOTE on impl for Bar
    //~| NOTE required by
}
