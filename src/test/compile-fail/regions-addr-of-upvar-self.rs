// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::uint;

struct dog {
    food: uint,
}

pub impl dog {
    fn chase_cat(&mut self) {
        for uint::range(0u, 10u) |_i| {
            let p: &'static mut uint = &mut self.food; //~ ERROR cannot infer an appropriate lifetime due to conflicting requirements
            *p = 3u;
        }
    }
}

fn main() {
}
