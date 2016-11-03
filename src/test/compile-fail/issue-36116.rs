// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo<T> {
    _a: T,
}

fn main() {
    let f = Some(Foo { _a: 42 }).map(|a| a as Foo::<i32>);
    //~^ ERROR unexpected token: `::`
    //~| HELP use `<...>` instead of `::<...>` if you meant to specify type arguments

    let g: Foo::<i32> = Foo { _a: 42 };
    //~^ ERROR unexpected token: `::`
    //~| HELP use `<...>` instead of `::<...>` if you meant to specify type arguments
}
