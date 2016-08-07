// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

impl (u8, u8) { //~ ERROR E0118
//~^ NOTE impl requires a base type
//~| NOTE either implement a trait on it or create a newtype to wrap it instead
    fn get_state(&self) -> String {
        String::new()
    }
}

fn main() {
}
