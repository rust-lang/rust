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
        x: 1,
        //~^ ERROR mismatched types
        //~| expected f32, found integral variable
        y: 2,
        //~^ ERROR mismatched types
        //~| expected f32, found integral variable
    };

    let pt2 = Point::<f32> {
        x: 3,
        //~^ ERROR mismatched types
        //~| expected f32, found integral variable
        y: 4,
        //~^ ERROR mismatched types
        //~| expected f32, found integral variable
    };

    let pair = PairF {
        x: 5,
        //~^ ERROR mismatched types
        //~| expected f32, found integral variable
        y: 6,
    };

    let pair2 = PairF::<i32> {
        x: 7,
        //~^ ERROR mismatched types
        //~| expected f32, found integral variable
        y: 8,
    };

    let pt3 = PointF::<i32> { //~ ERROR wrong number of type arguments
        x: 9,  //~ ERROR mismatched types
        y: 10, //~ ERROR mismatched types
    };

    match (Point { x: 1, y: 2 }) {
        PointF::<u32> { .. } => {} //~ ERROR wrong number of type arguments
        //~^ ERROR mismatched types
    }

    match (Point { x: 1, y: 2 }) {
        PointF { .. } => {} //~ ERROR mismatched types
    }

    match (Point { x: 1.0, y: 2.0 }) {
        PointF { .. } => {} // ok
    }

    match (Pair { x: 1, y: 2 }) {
        PairF::<u32> { .. } => {} //~ ERROR mismatched types
    }

    match (Pair { x: 1.0, y: 2 }) {
        PairF::<u32> { .. } => {} // ok
    }
}
