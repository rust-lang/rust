// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(exceeding_bitshifts)]

fn main() {
    let fat : [u8; (1<<61)+(1<<31)] =
        //~^ ERROR array length constant evaluation error: attempted left shift with overflow
        [0; (1u64<<61) as usize +(1u64<<31) as usize];
    //~^ ERROR expected constant integer for repeat count, but attempted left shift with overflow
}
