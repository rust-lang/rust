// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::io::{self, Read};
use std::vec;

pub struct Container<'a> {
    reader: &'a mut Read
}

impl<'a> Container<'a> {
    pub fn wrap<'s>(reader: &'s mut io::Read) -> Container<'s> {
        Container { reader: reader }
    }

    pub fn read_to(&mut self, vec: &mut [u8]) {
        self.reader.read(vec);
    }
}

pub fn for_stdin<'a>() -> Container<'a> {
    let mut r = io::stdin();
    Container::wrap(&mut r as &mut io::Read)
}

fn main() {
    let mut c = for_stdin();
    let mut v = Vec::new();
    c.read_to(v); //~ ERROR E0308
}
