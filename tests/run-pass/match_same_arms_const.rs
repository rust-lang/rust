// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(clippy::match_same_arms)]

const PRICE_OF_SWEETS: u32 = 5;
const PRICE_OF_KINDNESS: u32 = 0;
const PRICE_OF_DRINKS: u32 = 5;

pub fn price(thing: &str) -> u32 {
    match thing {
        "rolo" => PRICE_OF_SWEETS,
        "advice" => PRICE_OF_KINDNESS,
        "juice" => PRICE_OF_DRINKS,
        _ => panic!(),
    }
}

fn main() {}
