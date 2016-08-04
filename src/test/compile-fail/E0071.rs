// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo { FirstValue(i32) }

fn main() {
    let u = Foo::FirstValue { value: 0 };
    //~^ ERROR `Foo::FirstValue` does not name a struct or a struct variant [E0071]
    //~| NOTE not a struct

    let t = u32 { value: 4 };
    //~^ ERROR `u32` does not name a struct or a struct variant [E0071]
    //~| NOTE not a struct
}
