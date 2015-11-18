// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


fn main() {
    const ARR: [i32; 6] = [42, 43, 44, 45, 46, 47];
    const IDX: usize = 3;
    const VAL: i32 = ARR[IDX];
    const BLUB: [i32; (ARR[0] - 41) as usize] = [5];
}
