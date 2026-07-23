//@ edition:2021

#![feature(rustc_attrs)]

// Test to ensure that min analysis meets capture kind for all paths captured.

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let mut p = Point { x: 10, y: 20 };

    //
    // Requirements:
    // p.x -> MutBoorrow
    // p   -> Immutable
    //
    // Requirements met when p is captured via MutBorrow
    //
    let mut c = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        p.x += 10;
        //~^ NOTE: Capturing p[(0, 0)] -> Mutable
        //~| NOTE: p[] captured as Mutable here
        println!("{:?}", p);
        //~^ NOTE: Capturing p[] -> Immutable
        //~| NOTE: Min Capture p[] -> Mutable
        //~| NOTE: p[] used here
    };

    c();
}
