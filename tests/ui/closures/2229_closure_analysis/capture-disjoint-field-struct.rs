//@ edition:2021

#![feature(rustc_attrs)]

struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut p = Point { x: 10, y: 10 };

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR First Pass analysis includes:
    //~| ERROR Min Capture analysis includes:
        println!("{}", p.x);
        //~^ NOTE: Capturing p[(0, 0)] -> Immutable
        //~| NOTE: Min Capture p[(0, 0)] -> Immutable
    };

    // `c` should only capture `p.x`, therefore mutating `p.y` is allowed.
    let py = &mut p.y;

    c();
    *py = 20;
}
