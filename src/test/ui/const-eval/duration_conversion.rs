// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass

#![feature(duration_getters)]

use std::time::Duration;

fn main() {
    const _ONE_SECOND: Duration = Duration::from_nanos(1_000_000_000);
    const _ONE_MILLISECOND: Duration = Duration::from_nanos(1_000_000);
    const _ONE_MICROSECOND: Duration = Duration::from_nanos(1_000);
    const _ONE_NANOSECOND: Duration = Duration::from_nanos(1);
    const _ONE: usize = _ONE_SECOND.as_secs() as usize;
    const _TWO: usize = _ONE_MILLISECOND.subsec_millis() as usize;
    const _THREE: usize = _ONE_MICROSECOND.subsec_micros() as usize;
    const _FOUR: usize = _ONE_NANOSECOND.subsec_nanos() as usize;
    const _0: [[u8; _ONE]; _TWO] = [[1; _ONE]; _TWO];
    const _1: [[u8; _THREE]; _FOUR] = [[3; _THREE]; _FOUR];
}
