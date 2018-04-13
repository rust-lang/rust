// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

// Check that do not allow access to fields of uninitialized or moved
// structs.

#[derive(Default)]
struct Point {
    x: isize,
    y: isize,
}

#[derive(Default)]
struct Line {
    origin: Point,
    middle: Point,
    target: Point,
}

impl Line { fn consume(self) { } }

fn main() {
    let mut a: Point;
    let _ = a.x + 1; //[ast]~ ERROR use of possibly uninitialized variable: `a.x`
                     //[mir]~^ ERROR [E0381]

    let mut line1 = Line::default();
    let _moved = line1.origin;
    let _ = line1.origin.x + 1; //[ast]~ ERROR use of moved value: `line1.origin.x`
                                //[mir]~^ [E0382]

    let mut line2 = Line::default();
    let _moved = (line2.origin, line2.middle);
    line2.consume(); //[ast]~ ERROR use of partially moved value: `line2` [E0382]
                     //[mir]~^ [E0382]
}
