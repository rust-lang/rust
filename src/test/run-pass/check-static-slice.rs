// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the various ways of getting to a reference to a vec (both sized
// and unsized) work properly.

static aa: [int, ..3] = [1, 2, 3];
static ab: &'static [int, ..3] = &aa;
static ac: &'static [int] = ab;
static ad: &'static [int] = &aa;
static ae: &'static [int, ..3] = &[1, 2, 3];
static af: &'static [int] = &[1, 2, 3];

static ca: int = aa[0];
static cb: int = ab[1];
static cc: int = ac[2];
static cd: int = ad[0];
static ce: int = ae[1];
static cf: int = af[2];

fn main () {
    let b: &[int] = &[1, 2, 3];
    assert!(ac == b);
    assert!(ad == b);
    assert!(af == b);

    assert!(ca == 1);
    assert!(cb == 2);
    assert!(cc == 3);
    assert!(cd == 1);
    assert!(ce == 2);
    assert!(cf == 3);
}
