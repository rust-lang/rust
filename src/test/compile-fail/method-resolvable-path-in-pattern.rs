// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo;

trait MyTrait {
    fn trait_bar() {}
}

impl MyTrait for Foo {}

fn main() {
    match 0u32 {
        <Foo as MyTrait>::trait_bar => {}
        //~^ ERROR expected unit struct/variant or constant, found method `MyTrait::trait_bar`
    }
}
