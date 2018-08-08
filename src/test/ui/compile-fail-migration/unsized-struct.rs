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

struct Foo<T> { data: T }
fn foo1<T>() { not_sized::<Foo<T>>() } // Hunky dory.
fn foo2<T: ?Sized>() { not_sized::<Foo<T>>() }
//~^ ERROR the size for values of type
//
// Not OK: `T` is not sized.

struct Bar<T: ?Sized> { data: T }
fn bar1<T: ?Sized>() { not_sized::<Bar<T>>() }
fn bar2<T: ?Sized>() { is_sized::<Bar<T>>() }
//~^ ERROR the size for values of type
//
// Not OK: `Bar<T>` is not sized, but it should be.

fn main() { }
