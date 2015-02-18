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

    // This should cause a bounds-check panic, but may not if we do our
    // bounds checking by comparing the scaled index to the vector's
    // address-bounds, since we've scaled the index to wrap around to the
    // address of the 0th cell in the array (even though the index is
    // huge).

    let x = vec!(1_usize,2_usize,3_usize);

    let base = x.as_ptr() as uint;
    let idx = base / mem::size_of::<uint>();
    println!("ov1 base = 0x{:x}", base);
    println!("ov1 idx = 0x{:x}", idx);
    println!("ov1 sizeof::<uint>() = 0x{:x}", mem::size_of::<uint>());
    println!("ov1 idx * sizeof::<uint>() = 0x{:x}",
           idx * mem::size_of::<uint>());

    // This should panic.
    println!("ov1 0x{:x}", x[idx]);
}
