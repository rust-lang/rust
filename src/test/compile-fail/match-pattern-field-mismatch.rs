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
        rgb(usize, usize, usize),
        cmyk(usize, usize, usize, usize),
        no_color,
    }

    fn foo(c: color) {
        match c {
          color::rgb(_, _) => { }
          //~^ ERROR this pattern has 2 fields, but the corresponding tuple variant has 3 fields
          color::cmyk(_, _, _, _) => { }
          color::no_color => { }
        }
    }
}
