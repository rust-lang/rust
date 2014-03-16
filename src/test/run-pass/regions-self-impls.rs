// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Clam<'a> {
    chowder: &'a int
}

trait get_chowder<'a> {
    fn get_chowder(&self) -> &'a int;
}

impl<'a> get_chowder<'a> for Clam<'a> {
    fn get_chowder(&self) -> &'a int { return self.chowder; }
}

pub fn main() {
    let clam = Clam { chowder: &3 };
    println!("{:?}", *clam.get_chowder());
    clam.get_chowder();
}
