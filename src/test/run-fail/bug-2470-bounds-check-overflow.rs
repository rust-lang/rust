// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:index out of bounds

use std::mem;

fn main() {

    // This should cause a bounds-check failure, but may not if we do our
    // bounds checking by comparing the scaled index to the vector's
    // address-bounds, since we've scaled the index to wrap around to the
    // address of the 0th cell in the array (even though the index is
    // huge).

    let x = ~[1u,2u,3u];
    do x.as_imm_buf |p, _len| {
        let base = p as uint;
        let idx = base / mem::size_of::<uint>();
        error2!("ov1 base = 0x{:x}", base);
        error2!("ov1 idx = 0x{:x}", idx);
        error2!("ov1 sizeof::<uint>() = 0x{:x}", mem::size_of::<uint>());
        error2!("ov1 idx * sizeof::<uint>() = 0x{:x}",
               idx * mem::size_of::<uint>());

        // This should fail.
        error2!("ov1 0x{:x}",  x[idx]);
    }
}
