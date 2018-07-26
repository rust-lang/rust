// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cloudabi no std::fs support

#![allow(dead_code)]
#![deny(unused_imports)]

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
//~^ ERROR unused import: `BufRead`

pub fn read_from_file(path: &str) {
    let file = File::open(&path).unwrap();
    let mut reader = BufReader::new(file);
    let mut s = String::new();
    reader.read_to_string(&mut s).unwrap();
}

pub fn read_lines(s: &str) {
    for _line in s.lines() {

    }
}

fn main() {}
