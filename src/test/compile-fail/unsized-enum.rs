// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn is_sized<T:Sized>() { }
fn not_sized<T: ?Sized>() { }

enum Foo<U> { FooSome(U), FooNone }
fn foo1<T>() { not_sized::<Foo<T>>() } // Hunky dory.
fn foo2<T: ?Sized>() { not_sized::<Foo<T>>() }
//~^ ERROR `T: std::marker::Sized` is not satisfied
//
// Not OK: `T` is not sized.

fn main() { }
