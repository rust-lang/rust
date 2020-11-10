// FIXME(arora-aman) add run-pass once 2229 is implemented

#![feature(capture_disjoint_fields)]
//~^ WARNING the feature `capture_disjoint_fields` is incomplete
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
    // p   -> ImmBorrow
    //
    // Requirements met when p is captured via MutBorrow
    //
    let mut c = #[rustc_capture_analysis]
        //~^ ERROR: attributes on expressions are experimental
    || {
        p.x += 10;
        //~^ ERROR: Capturing p[(0, 0)] -> MutBorrow
        //~| ERROR: Min Capture p[] -> MutBorrow
        println!("{:?}", p);
        //~^ ERROR: Capturing p[] -> ImmBorrow
    };

    c();
}
