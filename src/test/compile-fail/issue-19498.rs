// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::A; //~ NOTE previous import of `A` here
use self::B; //~ NOTE previous import of `B` here
mod A {} //~ ERROR a module named `A` has already been imported in this module
//~| `A` already imported
pub mod B {} //~ ERROR a module named `B` has already been imported in this module
//~| `B` already imported
mod C {
    use C::D; //~ NOTE previous import of `D` here
    mod D {} //~ ERROR a module named `D` has already been imported in this module
    //~| `D` already imported
}

fn main() {}
