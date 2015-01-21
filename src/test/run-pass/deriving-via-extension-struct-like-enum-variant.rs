// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(PartialEq, Show)]
enum S {
    X { x: int, y: int },
    Y
}

pub fn main() {
    let x = S::X { x: 1, y: 2 };
    assert_eq!(x, x);
    assert!(!(x != x));
}
