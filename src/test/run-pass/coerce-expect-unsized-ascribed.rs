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

use std::fmt::Debug;

// A version of coerce-expect-unsized that uses type ascription.

pub fn main() {
    let _ = box { [1, 2, 3] }: Box<[int]>;
    let _ = box if true { [1, 2, 3] } else { [1, 3, 4] }: Box<[int]>;
    let _ = box match true { true => [1, 2, 3], false => [1, 3, 4] }: Box<[int]>;
    let _ = box { |x| (x as u8) }: Box<Fn(int) -> _>;
    let _ = box if true { false } else { true }: Box<Debug>;
    let _ = box match true { true => 'a', false => 'b' }: Box<Debug>;

    let _ = &{ [1, 2, 3] }: &[int];
    let _ = &if true { [1, 2, 3] } else { [1, 3, 4] }: &[int];
    let _ = &match true { true => [1, 2, 3], false => [1, 3, 4] }: &[int];
    let _ = &{ |x| (x as u8) }: &Fn(int) -> _;
    let _ = &if true { false } else { true }: &Debug;
    let _ = &match true { true => 'a', false => 'b' }: &Debug;

    let _ = Box::new([1, 2, 3]): Box<[int]>;
    let _ = Box::new(|x| (x as u8)): Box<Fn(int) -> _>;

    let _ = vec![
        Box::new(|x| (x as u8)),
        box |x| (x as i16 as u8),
    ]: Vec<Box<Fn(int) -> _>>;
}
