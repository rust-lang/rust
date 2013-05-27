// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    enum color {
        rgb(uint, uint, uint),
        cmyk(uint, uint, uint, uint),
        no_color,
    }

    fn foo(c: color) {
        match c {
          rgb(_, _, _) => { }
          cmyk(_, _, _, _) => { }
          no_color(_) => { }
          //~^ ERROR this pattern has 1 field, but the corresponding variant has no fields
        }
    }
}
