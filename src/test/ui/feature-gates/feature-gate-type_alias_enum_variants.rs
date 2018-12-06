// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo {
    Bar(i32),
    Baz { i: i32 },
}

type Alias = Foo;

fn main() {
    let t = Alias::Bar(0);
    //~^ ERROR enum variants on type aliases are experimental
    let t = Alias::Baz { i: 0 };
    //~^ ERROR enum variants on type aliases are experimental
    match t {
        Alias::Bar(_i) => {}
        //~^ ERROR enum variants on type aliases are experimental
        Alias::Baz { i: _i } => {}
        //~^ ERROR enum variants on type aliases are experimental
    }
}
