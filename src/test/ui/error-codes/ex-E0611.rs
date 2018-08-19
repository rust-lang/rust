// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {
    pub struct Foo(u32);

    impl Foo {
        pub fn new() -> Foo { Foo(0) }
    }
}

fn main() {
   let y = a::Foo::new();
   y.0; //~ ERROR field `0` of struct `a::Foo` is private
}
