// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn select<'r>(x: &'r int, y: &'r int) -> &'r int { x }

fn with<T>(f: |x: &int| -> T) -> T {
    f(&20)
}

fn manip<'a>(x: &'a int) -> int {
    let z = with(|y| { select(x, y) });
    //~^ ERROR cannot infer
    *z
}

fn main() {
}
