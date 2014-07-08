// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![allow(dead_code)]
#![deny(uppercase_variables)]

use std::io::File;
use std::io::IoError;

struct Something {
    X: uint //~ ERROR structure field names should start with a lowercase character
}

fn test(Xx: uint) { //~ ERROR variable names should start with a lowercase character
    println!("{}", Xx);
}

fn main() {
    let Test: uint = 0; //~ ERROR variable names should start with a lowercase character
    println!("{}", Test);

    let mut f = File::open(&Path::new("something.txt"));
    let mut buff = [0u8, ..16];
    match f.read(buff) {
        Ok(cnt) => println!("read this many bytes: {}", cnt),
        Err(IoError{ kind: EndOfFile, .. }) => println!("Got end of file: {}", EndOfFile.to_string()),
                        //~^ ERROR variable names should start with a lowercase character
    }

    test(1);

    let _ = Something { X: 0 };
}

