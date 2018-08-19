// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    const z: &'static isize = {
        //~^ ERROR let bindings in constants are unstable
        //~| ERROR statements in constants are unstable
        let p = 3;
        //~^ ERROR let bindings in constants are unstable
        //~| ERROR statements in constants are unstable
        &p //~ ERROR `p` does not live long enough
        //~^ ERROR let bindings in constants are unstable
    };
}
