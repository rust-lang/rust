// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test bounds checking for DST raw slices
// error-pattern:index out of bounds

fn main() {
    let a: *const [_] = &[1i, 2, 3];
    unsafe {
        let _b = (*a)[3];
    }
}
