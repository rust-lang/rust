// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:two_macros.rs
// ignore-stage1

#[macro_use]
#[macro_use()]
extern crate two_macros;

pub fn main() {
    macro_one!();
    macro_two!();
}
