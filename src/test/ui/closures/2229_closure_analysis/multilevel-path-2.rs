// FIXME(arora-aman) add run-pass once 2229 is implemented

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
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
        //~^ ERROR: attributes on expressions are experimental
        //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("{}", w.p.x);
        //~^ NOTE: Capturing w[(0, 0),(0, 0)] -> ImmBorrow
        //~| NOTE: Min Capture w[(0, 0),(0, 0)] -> ImmBorrow
    };

    // `c` only captures `w.p.x`, therefore it's safe to mutate `w.p.y`.
    let py = &mut w.p.y;
    c();

    *py = 20
}
