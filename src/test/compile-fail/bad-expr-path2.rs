// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod m1 {
    pub mod arguments {}
}

fn main(arguments: Vec<String>) { //~ ERROR main function has wrong type
    log(debug, m1::arguments);
    //~^ ERROR cannot find function `log` in this scope
    //~| ERROR cannot find value `debug` in this scope
    //~| ERROR expected value, found module `m1::arguments`
}
