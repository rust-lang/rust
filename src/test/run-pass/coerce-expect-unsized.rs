// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

use std::fmt::Show;

// Check that coercions apply at the pointer level and don't cause
// rvalue expressions to be unsized. See #20169 for more information.

pub fn main() {
    let _: Box<[int]> = box { [1, 2, 3] };
    let _: Box<[int]> = box if true { [1, 2, 3] } else { [1, 3, 4] };
    let _: Box<[int]> = box match true { true => [1, 2, 3], false => [1, 3, 4] };
    let _: Box<Fn(int) -> _> = box { |x| (x as u8) };
    let _: Box<Show> = box if true { false } else { true };
    let _: Box<Show> = box match true { true => 'a', false => 'b' };

    let _: &[int] = &{ [1, 2, 3] };
    let _: &[int] = &if true { [1, 2, 3] } else { [1, 3, 4] };
    let _: &[int] = &match true { true => [1, 2, 3], false => [1, 3, 4] };
    let _: &Fn(int) -> _ = &{ |x| (x as u8) };
    let _: &Show = &if true { false } else { true };
    let _: &Show = &match true { true => 'a', false => 'b' };
}
