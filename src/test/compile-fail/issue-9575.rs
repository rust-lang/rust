// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(start)]

#[start]
fn start(argc: isize, argv: *const *const u8, crate_map: *const u8) -> isize {
    //~^ ERROR incorrect number of function parameters
   0
}
