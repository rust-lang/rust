// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum color { rgb(isize, isize, isize), rgba(isize, isize, isize, isize), }

fn main() {
    let red: color = rgb(255, 0, 0); //~ ERROR unresolved name `rgb`
    match red {
      rgb(r, g, b) => { println!("rgb"); } //~ ERROR unresolved enum variant, struct or const `rgb`
      hsl(h, s, l) => { println!("hsl"); } //~ ERROR unresolved enum variant, struct or const `hsl`
    }
}
