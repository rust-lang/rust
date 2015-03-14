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
        //~^ ERROR structure constructor specifies a structure of type
        //~| expected f32
        //~| found integral variable
        x: 1,
        y: 2,
    };

    let pt2 = Point::<f32> {
        //~^ ERROR structure constructor specifies a structure of type
        //~| expected f32
        //~| found integral variable
        x: 3,
        y: 4,
    };

    let pair = PairF {
        //~^ ERROR structure constructor specifies a structure of type
        //~| expected f32
        //~| found integral variable
        x: 5,
        y: 6,
    };

    let pair2 = PairF::<i32> {
        //~^ ERROR structure constructor specifies a structure of type
        //~| expected f32
        //~| found integral variable
        x: 7,
        y: 8,
    };

    let pt3 = PointF::<i32> {
        //~^ ERROR wrong number of type arguments
        //~| ERROR structure constructor specifies a structure of type
        x: 9,
        y: 10,
    };
}
