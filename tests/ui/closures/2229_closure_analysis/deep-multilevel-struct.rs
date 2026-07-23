//@ edition:2021

#![feature(rustc_attrs)]
#![allow(unused)]

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}
#[derive(Debug)]
struct Line {
    p: Point,
    q: Point
}
#[derive(Debug)]
struct Plane {
    a: Line,
    b: Line,
}

fn main() {
    let mut p = Plane {
        a: Line {
            p: Point { x: 1,y: 2 },
            q: Point { x: 3,y: 4 },
        },
        b: Line {
            p: Point { x: 1,y: 2 },
            q: Point { x: 3,y: 4 },
        }
    };

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        let x = &p.a.p.x;
        //~^ NOTE: Capturing p[(0, 0),(0, 0),(0, 0)] -> Immutable
        p.b.q.y = 9;
        //~^ NOTE: Capturing p[(1, 0),(1, 0),(1, 0)] -> Mutable
        //~| NOTE: p[] captured as Mutable here
        println!("{:?}", p);
        //~^ NOTE: Capturing p[] -> Immutable
        //~| NOTE: Min Capture p[] -> Mutable
        //~| NOTE: p[] used here
    };
}
