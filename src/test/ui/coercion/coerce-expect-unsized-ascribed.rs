// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// A version of coerce-expect-unsized that uses type ascription.
// Doesn't work so far, but supposed to work eventually

#![feature(box_syntax, type_ascription)]

use std::fmt::Debug;

pub fn main() {
    let _ = box { [1, 2, 3] }: Box<[i32]>; //~ ERROR mismatched types
    let _ = box if true { [1, 2, 3] } else { [1, 3, 4] }: Box<[i32]>; //~ ERROR mismatched types
    let _ = box match true { true => [1, 2, 3], false => [1, 3, 4] }: Box<[i32]>;
    //~^ ERROR mismatched types
    let _ = box { |x| (x as u8) }: Box<Fn(i32) -> _>; //~ ERROR mismatched types
    let _ = box if true { false } else { true }: Box<Debug>; //~ ERROR mismatched types
    let _ = box match true { true => 'a', false => 'b' }: Box<Debug>; //~ ERROR mismatched types

    let _ = &{ [1, 2, 3] }: &[i32]; //~ ERROR mismatched types
    let _ = &if true { [1, 2, 3] } else { [1, 3, 4] }: &[i32]; //~ ERROR mismatched types
    let _ = &match true { true => [1, 2, 3], false => [1, 3, 4] }: &[i32];
    //~^ ERROR mismatched types
    let _ = &{ |x| (x as u8) }: &Fn(i32) -> _; //~ ERROR mismatched types
    let _ = &if true { false } else { true }: &Debug; //~ ERROR mismatched types
    let _ = &match true { true => 'a', false => 'b' }: &Debug; //~ ERROR mismatched types

    let _ = Box::new([1, 2, 3]): Box<[i32]>; //~ ERROR mismatched types
    let _ = Box::new(|x| (x as u8)): Box<Fn(i32) -> _>; //~ ERROR mismatched types

    let _ = vec![
        Box::new(|x| (x as u8)),
        box |x| (x as i16 as u8),
    ]: Vec<Box<Fn(i32) -> _>>;
}
