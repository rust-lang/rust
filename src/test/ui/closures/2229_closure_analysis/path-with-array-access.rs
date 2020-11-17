#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| NOTE: `#[warn(incomplete_features)]` on by default
//~| NOTE: see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
#![feature(rustc_attrs)]

struct Point {
    x: f32,
    y: f32,
}

struct Pentagon {
    points: [Point; 5],
}

fn main() {
    let p1 = Point { x: 10.0, y: 10.0 };
    let p2 = Point { x: 7.5, y: 12.5 };
    let p3 = Point { x: 15.0, y: 15.0 };
    let p4 = Point { x: 12.5, y: 12.5 };
    let p5 = Point { x: 20.0, y: 10.0 };

    let pent = Pentagon { points: [p1, p2, p3, p4, p5] };

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("{}", pent.points[5].x);
        //~^ NOTE: Capturing pent[(0, 0)] -> ImmBorrow
        //~| NOTE: Min Capture pent[(0, 0)] -> ImmBorrow
    };
}
