// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::slice::Iter;
use std::io::{Error, ErrorKind, Result};
use std::vec::*;

fn foo(it: &mut Iter<u8>) -> Result<u8> {
    Ok(*it.next().unwrap())
}

fn bar() -> Result<u8> {
    let data: Vec<u8> = Vec::new();

    if true {
        return Err(Error::new(ErrorKind::NotFound, "msg"));
    }

    let mut it = data.iter();
    foo(&mut it)
}

fn main() {
    bar();
}
