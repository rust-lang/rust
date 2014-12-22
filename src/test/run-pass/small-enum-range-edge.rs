// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
 * Tests the range assertion wraparound case in trans::middle::adt::load_discr.
 */

#[repr(u8)]
enum Eu { Lu = 0, Hu = 255 }

impl Copy for Eu {}

static CLu: Eu = Eu::Lu;
static CHu: Eu = Eu::Hu;

#[repr(i8)]
enum Es { Ls = -128, Hs = 127 }

impl Copy for Es {}

static CLs: Es = Es::Ls;
static CHs: Es = Es::Hs;

pub fn main() {
    assert_eq!((Eu::Hu as u8) + 1, Eu::Lu as u8);
    assert_eq!((Es::Hs as i8) + 1, Es::Ls as i8);
    assert_eq!(CLu as u8, Eu::Lu as u8);
    assert_eq!(CHu as u8, Eu::Hu as u8);
    assert_eq!(CLs as i8, Es::Ls as i8);
    assert_eq!(CHs as i8, Es::Hs as i8);
}
