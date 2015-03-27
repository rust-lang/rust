// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// issues #10618 and #16382
// pretty-expanded FIXME #23616

const SIZE: isize = 25;

fn main() {
    let _a: [bool; 1 as usize];
    let _b: [isize; SIZE as usize] = [1; SIZE as usize];
    let _c: [bool; '\n' as usize] = [true; '\n' as usize];
    let _d: [bool; true as usize] = [true; true as usize];
}
