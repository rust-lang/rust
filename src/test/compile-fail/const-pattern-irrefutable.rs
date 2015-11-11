// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod foo {
    const b: u8 = 2; //~ NOTE constant defined here
    const d: u8 = 2; //~ NOTE constant defined here
}

use foo::b as c; //~ NOTE constant imported here
use foo::d; //~ NOTE constant imported here

const a: u8 = 2; //~ NOTE constant defined here

fn main() {
    let a = 4; //~ ERROR only irrefutable
               //~^ NOTE there already is a constant in scope
    let c = 4; //~ ERROR only irrefutable
               //~^ NOTE there already is a constant in scope
    let d = 4; //~ ERROR only irrefutable
               //~^ NOTE there already is a constant in scope
}
