// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ptr;

pub fn main() {
    let x = ~1;
    let y = ptr::to_unsafe_ptr(&(*x)) as uint;
    let lam_move: @fn() -> uint = || ptr::to_unsafe_ptr(&(*x)) as uint;
    assert_eq!(lam_move(), y);

    let x = ~2;
    let y = ptr::to_unsafe_ptr(&(*x)) as uint;
    let lam_move: @fn() -> uint = || ptr::to_unsafe_ptr(&(*x)) as uint;
    assert_eq!(lam_move(), y);

    let x = ~3;
    let y = ptr::to_unsafe_ptr(&(*x)) as uint;
    let snd_move: ~fn() -> uint = || ptr::to_unsafe_ptr(&(*x)) as uint;
    assert_eq!(snd_move(), y);

    let x = ~4;
    let y = ptr::to_unsafe_ptr(&(*x)) as uint;
    let lam_move: ~fn() -> uint = || ptr::to_unsafe_ptr(&(*x)) as uint;
    assert_eq!(lam_move(), y);
}
