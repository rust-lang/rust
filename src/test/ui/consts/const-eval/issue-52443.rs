// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    [(); & { loop { continue } } ]; //~ ERROR mismatched types
    [(); loop { break }]; //~ ERROR mismatched types
    [(); {while true {break}; 0}]; //~ ERROR constant contains unimplemented expression type
    [(); { for _ in 0usize.. {}; 0}]; //~ ERROR calls in constants are limited to constant functions
    //~^ ERROR constant contains unimplemented expression type
    //~| ERROR could not evaluate repeat length
}
