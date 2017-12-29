// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// FIXME (#23926): the error output is not consistent between a
// self-hosted and a cross-compiled setup. Skipping for now.

// ignore-test FIXME(#23926)

#![allow(exceeding_bitshifts)]

fn main() {
    let _fat : [u8; (1<<61)+(1<<31)] =
        [0; (1u64<<61) as usize +(1u64<<31) as usize];
}
