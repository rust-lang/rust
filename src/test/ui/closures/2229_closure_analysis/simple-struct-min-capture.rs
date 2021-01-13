// FIXME(arora-aman) add run-pass once 2229 is implemented

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
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
        //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        p.x += 10;
        //~^ NOTE: Capturing p[(0, 0)] -> MutBorrow
        //~| NOTE: Min Capture p[] -> MutBorrow
        println!("{:?}", p);
        //~^ NOTE: Capturing p[] -> ImmBorrow
    };

    c();
}
