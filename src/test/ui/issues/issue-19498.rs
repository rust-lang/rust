// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::A;
use self::B;
mod A {} //~ ERROR the name `A` is defined multiple times
//~| `A` redefined here
pub mod B {} //~ ERROR the name `B` is defined multiple times
//~| `B` redefined here
mod C {
    use C::D;
    mod D {} //~ ERROR the name `D` is defined multiple times
    //~| `D` redefined here
}

fn main() {}
