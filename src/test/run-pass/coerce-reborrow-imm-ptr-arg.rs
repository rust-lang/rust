// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn negate(x: &int) -> int {
    -*x
}

fn negate_mut(y: &mut int) -> int {
    negate(y)
}

fn negate_imm(y: &int) -> int {
    negate(y)
}

pub fn main() {}
