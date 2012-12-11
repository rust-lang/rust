// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

type clam = { chowder: &int };

trait get_chowder {
    fn get_chowder() -> &self/int;
}

impl clam: get_chowder {
    fn get_chowder() -> &self/int { return self.chowder; }
}

fn main() {
    let clam = { chowder: &3 };
    log(debug, *clam.get_chowder());
    clam.get_chowder();
}

