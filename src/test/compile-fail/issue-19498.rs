// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use self::A; //~ ERROR import `A` conflicts with existing submodule
use self::B; //~ ERROR import `B` conflicts with existing submodule
mod A {}
pub mod B {}

mod C {
    use C::D; //~ ERROR import `D` conflicts with existing submodule
    mod D {}
}

fn main() {}
