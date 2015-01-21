// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:cci_const.rs

extern crate cci_const;
use cci_const::bar;
static foo: extern "C" fn() = bar;

pub fn main() {
    assert!(foo == bar);
}
