// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type Foo<T,T> = Option<T>;
//~^ ERROR the name `T` is already used

struct Bar<T,T>(T);
//~^ ERROR the name `T` is already used

struct Baz<T,T> {
//~^ ERROR the name `T` is already used
    x: T,
}

enum Boo<T,T> {
//~^ ERROR the name `T` is already used
    A(T),
    B,
}

fn quux<T,T>(x: T) {}
//~^ ERROR the name `T` is already used

trait Qux<T,T> {}
//~^ ERROR the name `T` is already used

impl<T,T> Qux<T,T> for Option<T> {}
//~^ ERROR the name `T` is already used
//~^^ ERROR the type parameter `T` is not constrained

fn main() {
}
