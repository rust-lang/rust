// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This is a regression test for #52967, where we discovered that in
// the initial deployment of NLL for the 2018 edition, I forgot to
// turn on two-phase-borrows in addition to `-Z borrowck=migrate`.

// revisions: ast zflags edition
//[zflags]compile-flags: -Z borrowck=migrate -Z two-phase-borrows
//[edition]compile-flags: --edition 2018

// run-pass

fn the_bug() {
    let mut stuff = ("left", "right");
    match stuff {
        (ref mut left, _) if *left == "left" => { *left = "new left"; }
        _ => {}
    }
    assert_eq!(stuff, ("new left", "right"));
}

fn main() {
    the_bug();
}
