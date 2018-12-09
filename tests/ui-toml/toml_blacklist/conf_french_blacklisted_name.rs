// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![allow(clippy::single_match)]
#![allow(unused_variables)]
#![warn(clippy::blacklisted_name)]

fn test(toto: ()) {}

fn main() {
    let toto = 42;
    let tata = 42;
    let titi = 42;

    let tatab = 42;
    let tatatataic = 42;

    match (42, Some(1337), Some(0)) {
        (toto, Some(tata), titi @ Some(_)) => (),
        _ => (),
    }
}
