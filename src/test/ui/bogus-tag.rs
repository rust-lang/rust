// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Color { Rgb(isize, isize, isize), Rgba(isize, isize, isize, isize), }

fn main() {
    let red: Color = Color::Rgb(255, 0, 0);
    match red {
      Color::Rgb(r, g, b) => { println!("rgb"); }
      Color::Hsl(h, s, l) => { println!("hsl"); }
      //~^ ERROR no variant
    }
}
