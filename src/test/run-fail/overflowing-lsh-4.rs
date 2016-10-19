// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:thread 'main' panicked at 'attempt to shift left with overflow'
// compile-flags: -C debug-assertions

// This function is checking that our automatic truncation does not
// sidestep the overflow checking.

#![warn(exceeding_bitshifts)]

fn main() {
    // this signals overflow when checking is on
    let x = 1_i8 << 17;

    // ... but when checking is off, the fallback will truncate the
    // input to its lower three bits (= 1). Note that this is *not*
    // the behavior of the x86 processor for 8- and 16-bit types,
    // but it is necessary to avoid undefined behavior from LLVM.
    //
    // We check that here, by ensuring the result has only been
    // shifted by one place; if overflow checking is turned off, then
    // this assertion will pass (and the compiletest driver will
    // report that the test did not produce the error expected above).
    assert_eq!(x, 2_i8);
}
