// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    use std::mem::replace;
    let mut x = 5i;
    replace(&mut x, 6);
    {
        use std::mem::*;
        let mut y = 6i;
        swap(&mut x, &mut y);
    }
}
