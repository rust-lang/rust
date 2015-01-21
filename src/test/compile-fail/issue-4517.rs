// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn bar(int_param: usize) {}

fn main() {
    let foo: [u8; 4] = [1u8; 4us];
    bar(foo);
    //~^ ERROR mismatched types
    //~| expected `usize`
    //~| found `[u8; 4]`
    //~| expected usize
    //~| found array of 4 elements
}
