// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod Bar {
    pub struct Foo {
        pub a: isize,
        b: isize,
    }

    pub struct FooTuple (
        pub isize,
        isize,
    );
}

fn pat_match(foo: Bar::Foo) {
    let Bar::Foo{a:a, b:b} = foo; //~ ERROR E0451
                                  //~^ NOTE field `b` is private
}

fn pat_match_tuple(foo: Bar::FooTuple) {
    let Bar::FooTuple(a,b) = foo; //~ ERROR E0451
                                  //~^ NOTE field `1` is private
}

fn main() {
    let f = Bar::Foo{ a: 0, b: 0 }; //~ ERROR E0451
                                    //~^ NOTE field `b` is private
}
