// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::iter::repeat;

fn main() {
    let mut vector = vec![1us, 2];
    for &x in vector.iter() {
        let cap = vector.capacity();
        vector.extend(repeat(0));      //~ ERROR cannot borrow
        vector[1us] = 5us;   //~ ERROR cannot borrow
    }
}

