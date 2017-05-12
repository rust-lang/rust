// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::Self as Foo; //~ ERROR unresolved import `self::Self`

pub fn main() {
    let Self = 5;
    //~^ ERROR cannot find unit struct/variant or constant `Self` in this scope

    match 15 {
        Self => (),
        //~^ ERROR cannot find unit struct/variant or constant `Self` in this scope
        Foo { x: Self } => (),
        //~^ ERROR cannot find unit struct/variant or constant `Self` in this scope
    }
}
