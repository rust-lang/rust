//@ edition:2021

#![feature(rustc_attrs)]
#![allow(unused)]

struct Point {
    x: i32,
    y: i32,
}
struct Wrapper {
    p: Point,
}

fn main() {
    let mut w = Wrapper { p: Point { x: 10, y: 10 } };

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("{}", w.p.x);
        //~^ NOTE: Capturing w[(0, 0),(0, 0)] -> Immutable
        //~| NOTE: Min Capture w[(0, 0),(0, 0)] -> Immutable
    };

    // `c` only captures `w.p.x`, therefore it's safe to mutate `w.p.y`.
    let py = &mut w.p.y;
    c();

    *py = 20
}
