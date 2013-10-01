// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:quux

struct faily_box {
    i: @int
}
// What happens to the box pointer owned by this class?

fn faily_box(i: @int) -> faily_box { faily_box { i: i } }

#[unsafe_destructor]
impl Drop for faily_box {
    fn drop(&mut self) {
        fail2!("quux");
    }
}

fn main() {
    faily_box(@10);
}
