// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::raw::Slice;

fn main() {
    match () {
        Slice { data: data, len: len } => (),
        //~^ ERROR mismatched types
        //~| expected `()`
        //~| found `core::raw::Slice<_>`
        //~| expected ()
        //~| found struct `core::raw::Slice`
        _ => unreachable!()
    }
}
