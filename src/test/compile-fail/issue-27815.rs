// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod A {}

fn main() {
    let u = A { x: 1 }; //~ ERROR `A` does not name a struct or a struct variant
    let v = u32 { x: 1 }; //~ ERROR `u32` does not name a struct or a struct variant
    match () {
        A { x: 1 } => {}
        //~^ ERROR expected variant, struct or type alias, found module `A`
        u32 { x: 1 } => {}
        //~^ ERROR expected variant, struct or type alias, found builtin type `u32`
    }
}
