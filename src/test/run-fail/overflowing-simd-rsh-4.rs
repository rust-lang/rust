// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:thread '<main>' panicked at 'shift operation overflowed'
// compile-flags: -C debug-assertions

// This function is checking that our (type-based) automatic
// truncation does not sidestep the overflow checking.

#![feature(core_simd)]

use std::simd::i8x16;

fn eq_i8x16(i8x16(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15): i8x16,
            i8x16(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15): i8x16)
            -> bool
{
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
        && (x4 == y4) && (x5 == y5) && (x6 == y6) && (x7 == y7)
        && (x8 == y8) && (x9 == y9) && (x10 == y10) && (x11 == y11)
        && (x12 == y12) && (x13 == y13) && (x14 == y14) && (x15 == y15)
}

// (Work around constant-evaluation)
fn id<T>(x: T) -> T { x }

fn main() {
    // this signals overflow when checking is on
    let x = i8x16(2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        >> id(i8x16(17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));

    // ... but when checking is off, the fallback will truncate the
    // input to its lower three bits (= 1). Note that this is *not*
    // the behavior of the x86 processor for 8- and 16-bit types,
    // but it is necessary to avoid undefined behavior from LLVM.
    //
    // We check that here, by ensuring the result is not zero; if
    // overflow checking is turned off, then this assertion will pass
    // (and the compiletest driver will report that the test did not
    // produce the error expected above).
    assert!(eq_i8x16(x, i8x16(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
}
