// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Clam<'self> {
    chowder: &'self int
}

trait get_chowder<'self> {
    fn get_chowder(&self) -> &'self int;
}

impl<'self> get_chowder<'self> for Clam<'self> {
    fn get_chowder(&self) -> &'self int { return self.chowder; }
}

pub fn main() {
    let clam = Clam { chowder: &3 };
    info!("{:?}", *clam.get_chowder());
    clam.get_chowder();
}
