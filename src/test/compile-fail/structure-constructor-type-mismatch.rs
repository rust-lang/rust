// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Point<T> {
    x: T,
    y: T,
}

type PointF = Point<f32>;

struct Pair<T,U> {
    x: T,
    y: U,
}

type PairF<U> = Pair<f32,U>;

fn main() {
    let pt = PointF {
        //~^ ERROR expected f32, found int
        x: 1i,
        y: 2i,
    };

    let pt2 = Point::<f32> {
        //~^ ERROR expected f32, found int
        x: 3i,
        y: 4i,
    };

    let pair = PairF {
        //~^ ERROR expected f32, found int
        x: 5i,
        y: 6i,
    };

    let pair2 = PairF::<int> {
        //~^ ERROR expected f32, found int
        x: 7i,
        y: 8i,
    };

    let pt3 = PointF::<int> {
        //~^ ERROR wrong number of type arguments
        x: 9i,
        y: 10i,
    };
}

