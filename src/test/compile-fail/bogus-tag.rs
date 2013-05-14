// -*- rust -*-
// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// error-pattern: unresolved

enum color { rgb(int, int, int), rgba(int, int, int, int), }

fn main() {
    let red: color = rgb(255, 0, 0);
    match red {
      rgb(r, g, b) => { debug!("rgb"); }
      hsl(h, s, l) => { debug!("hsl"); }
    }
}
