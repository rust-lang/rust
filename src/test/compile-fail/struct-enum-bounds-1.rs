// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that incorrect bounds in structs and enums are picked up.

struct Foo<X: T>; //~ERROR attempt to bound type parameter with a nonexistent trait `T`
enum Bar<X: T> {} //~ERROR attempt to bound type parameter with a nonexistent trait `T`

trait T2<X> {}

struct Baz<X: T2<T>>; //~ ERROR use of undeclared type name `T`
enum Qux<X: T2<T>> {} //~ ERROR use of undeclared type name `T`

fn main() {}
