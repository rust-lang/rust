// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod buildings {
    pub struct Tower { pub height: uint }
}

pub fn main() {
    let sears = buildings::Tower { height: 1451 };
    let h: uint = match sears {
        buildings::Tower { height: h } => { h }
    };

    println!("{}", h);
}
