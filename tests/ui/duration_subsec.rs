// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::duration_subsec)]

use std::time::Duration;

fn main() {
    let dur = Duration::new(5, 0);

    let bad_millis_1 = dur.subsec_micros() / 1_000;
    let bad_millis_2 = dur.subsec_nanos() / 1_000_000;
    let good_millis = dur.subsec_millis();
    assert_eq!(bad_millis_1, good_millis);
    assert_eq!(bad_millis_2, good_millis);

    let bad_micros = dur.subsec_nanos() / 1_000;
    let good_micros = dur.subsec_micros();
    assert_eq!(bad_micros, good_micros);

    // Handle refs
    let _ = (&dur).subsec_nanos() / 1_000;

    // Handle constants
    const NANOS_IN_MICRO: u32 = 1_000;
    let _ = dur.subsec_nanos() / NANOS_IN_MICRO;

    // Other literals aren't linted
    let _ = dur.subsec_nanos() / 699;
}
