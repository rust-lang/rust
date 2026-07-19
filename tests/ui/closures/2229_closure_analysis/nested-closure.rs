//@ edition:2021

#![feature(rustc_attrs)]

struct Point {
    x: i32,
    y: i32,
}

// This testcase ensures that nested closures are handles properly
// - The nested closure is analyzed first.
// - The capture kind of the nested closure is accounted for by the enclosing closure
// - Any captured path by the nested closure that starts off a local variable in the enclosing
// closure is not listed as a capture of the enclosing closure.

fn main() {
    let mut p = Point { x: 5, y: 20 };

    let mut c1 = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("{}", p.x);
        //~^ NOTE: Capturing p[(0, 0)] -> Immutable
        //~| NOTE: Min Capture p[(0, 0)] -> Immutable
        let incr = 10;
        let mut c2 = #[rustc_capture_analysis]
        || p.y += incr;
        //~^ ERROR: First Pass analysis includes:
        //~| ERROR: Min Capture analysis includes:
        //~| NOTE: Capturing p[(1, 0)] -> Mutable
        //~| NOTE: Capturing incr[] -> Immutable
        //~| NOTE: Min Capture p[(1, 0)] -> Mutable
        //~| NOTE: Min Capture incr[] -> Immutable
        //~| NOTE: Capturing p[(1, 0)] -> Mutable
        //~| NOTE: Min Capture p[(1, 0)] -> Mutable
        c2();
        println!("{}", p.y);
        //~^ NOTE: Capturing p[(1, 0)] -> Immutable
    };

    c1();

    let px = &p.x;

    println!("{}", px);

    c1();
}
