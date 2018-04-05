// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #47295: We used to have a hack of special-casing adjacent amtch
// arms whose patterns were composed solely of constants to not have
// them linked in the cfg.
//
// THis was broken for various reasons. In particular, that hack was
// originally authored under the assunption that other checks
// elsewhere would ensure that the two patterns did not overlap.  But
// that assumption did not hold, at least not in the long run (namely,
// overlapping patterns were turned into warnings rather than errors).

#![feature(box_syntax)]

fn main() {
    let x: Box<_> = box 1;

    let v = (1, 2);

    match v {
        (1, 2) if take(x) => (),
        (1, 2) if take(x) => (), //~ ERROR use of moved value: `x`
        _ => (),
    }
}

fn take<T>(_: T) -> bool { false }
