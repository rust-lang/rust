// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! columnline {
    () => (
        (column!(), line!())
    )
}

macro_rules! indirectcolumnline {
    () => (
        (||{ columnline!() })()
    )
}

fn main() {
    let closure = || {
        columnline!()
    };
    let iflet = if let Some(_) = Some(0) {
        columnline!()
    } else { (0, 0) };
    let cl = columnline!();
    assert_eq!(closure(), (9, 25));
    assert_eq!(iflet, (9, 28));
    assert_eq!(cl, (14, 30));
    let indirect = indirectcolumnline!();
    assert_eq!(indirect, (20, 34));
}
