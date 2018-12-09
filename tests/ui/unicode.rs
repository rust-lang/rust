// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::zero_width_space)]
fn zero() {
    print!("Here >​< is a ZWS, and ​another");
    print!("This\u{200B}is\u{200B}fine");
}

#[warn(clippy::unicode_not_nfc)]
fn canon() {
    print!("̀àh?");
    print!("a\u{0300}h?"); // also okay
}

#[warn(clippy::non_ascii_literal)]
fn uni() {
    print!("Üben!");
    print!("\u{DC}ben!"); // this is okay
}

fn main() {
    zero();
    uni();
    canon();
}
