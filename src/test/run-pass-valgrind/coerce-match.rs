// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that coercions are propagated through match and if expressions.

// pretty-expanded FIXME #23616

#![allow(unknown_features)]
#![feature(box_syntax)]

pub fn main() {
    let _: Box<[isize]> =
        if true { let b: Box<_> = box [1, 2, 3]; b } else { let b: Box<_> = box [1]; b };

    let _: Box<[isize]> = match true {
        true => { let b: Box<_> = box [1, 2, 3]; b }
        false => { let b: Box<_> = box [1]; b }
    };

    // Check we don't get over-keen at propagating coercions in the case of casts.
    let x = if true { 42 } else { 42u8 } as u16;
    let x = match true { true => 42, false => 42u8 } as u16;
}
