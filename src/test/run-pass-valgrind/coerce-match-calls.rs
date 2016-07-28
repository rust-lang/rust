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

use std::boxed::Box;

pub fn main() {
    let _: Box<[isize]> = if true { Box::new([1, 2, 3]) } else { Box::new([1]) };

    let _: Box<[isize]> = match true { true => Box::new([1, 2, 3]), false => Box::new([1]) };

    // Check we don't get over-keen at propagating coercions in the case of casts.
    let x = if true { 42 } else { 42u8 } as u16;
    let x = match true { true => 42, false => 42u8 } as u16;
}
