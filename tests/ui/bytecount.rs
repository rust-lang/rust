// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deny(clippy::naive_bytecount)]
fn main() {
    let x = vec![0_u8; 16];

    let _ = x.iter().filter(|&&a| a == 0).count(); // naive byte count

    let _ = (&x[..]).iter().filter(|&a| *a == 0).count(); // naive byte count

    let _ = x.iter().filter(|a| **a > 0).count(); // not an equality count, OK.

    let _ = x.iter().map(|a| a + 1).filter(|&a| a < 15).count(); // not a slice

    let b = 0;

    let _ = x.iter().filter(|_| b > 0).count(); // woah there

    let _ = x.iter().filter(|_a| b == b + 1).count(); // nothing to see here, move along

    let _ = x.iter().filter(|a| b + 1 == **a).count(); // naive byte count

    let y = vec![0_u16; 3];

    let _ = y.iter().filter(|&&a| a == 0).count(); // naive count, but not bytes
}
