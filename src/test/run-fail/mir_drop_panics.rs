// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:panic 1
// error-pattern:drop 2
use std::io::{self, Write};

struct Droppable(u32);
impl Drop for Droppable {
    fn drop(&mut self) {
        if self.0 == 1 {
            panic!("panic 1");
        } else {
            write!(io::stderr(), "drop {}", self.0);
        }
    }
}

fn mir() {
    let x = Droppable(2);
    let y = Droppable(1);
}

fn main() {
    mir();
}
