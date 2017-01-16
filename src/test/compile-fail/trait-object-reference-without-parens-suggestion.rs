// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let _: &Copy + 'static;
    //~^ ERROR expected a path
    //~| HELP try adding parentheses
    //~| SUGGESTION let _: &(Copy + 'static);
    //~| ERROR at least one non-builtin trait is required for an object type
    let _: &'static Copy + 'static;
    //~^ ERROR expected a path
    //~| HELP try adding parentheses
    //~| SUGGESTION let _: &'static (Copy + 'static);
    //~| ERROR at least one non-builtin trait is required for an object type
}
