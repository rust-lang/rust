// edition:2021

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
        //~^ ERROR: attributes on expressions are experimental
        //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("{}", p.x);
        //~^ NOTE: Capturing p[(0, 0)] -> ImmBorrow
        //~| NOTE: Min Capture p[(0, 0)] -> ImmBorrow
        let incr = 10;
        let mut c2 = #[rustc_capture_analysis]
        //~^ ERROR: attributes on expressions are experimental
        //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
        || p.y += incr;
        //~^ ERROR: First Pass analysis includes:
        //~| ERROR: Min Capture analysis includes:
        //~| NOTE: Capturing p[(1, 0)] -> MutBorrow
        //~| NOTE: Capturing incr[] -> ImmBorrow
        //~| NOTE: Min Capture p[(1, 0)] -> MutBorrow
        //~| NOTE: Min Capture incr[] -> ImmBorrow
        //~| NOTE: Capturing p[(1, 0)] -> MutBorrow
        //~| NOTE: Min Capture p[(1, 0)] -> MutBorrow
        c2();
        println!("{}", p.y);
        //~^ NOTE: Capturing p[(1, 0)] -> ImmBorrow
    };

    c1();

    let px = &p.x;

    println!("{}", px);

    c1();
}
