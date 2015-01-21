// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15877

pub fn main() {
    let x = 1i;
    let y = 2i;

    assert_eq!(3i, match (x, y) {
        (1, 1) => 1,
        (2, 2) => 2,
        (1...2, 2) => 3,
        _ => 4,
    });

    // nested tuple
    assert_eq!(3i, match ((x, y),) {
        ((1, 1),) => 1,
        ((2, 2),) => 2,
        ((1...2, 2),) => 3,
        _ => 4,
    });
}
