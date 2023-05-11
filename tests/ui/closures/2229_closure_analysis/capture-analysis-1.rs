// edition:2021

#![feature(rustc_attrs)]

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let p = Point { x: 10, y: 10 };
    let q = Point { x: 10, y: 10 };

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ First Pass analysis includes:
    //~| Min Capture analysis includes:
        println!("{:?}", p);
        //~^ NOTE: Capturing p[] -> ImmBorrow
        //~| NOTE: Min Capture p[] -> ImmBorrow
        println!("{:?}", p.x);
        //~^ NOTE: Capturing p[(0, 0)] -> ImmBorrow

        println!("{:?}", q.x);
        //~^ NOTE: Capturing q[(0, 0)] -> ImmBorrow
        println!("{:?}", q);
        //~^ NOTE: Capturing q[] -> ImmBorrow
        //~| NOTE: Min Capture q[] -> ImmBorrow
    };
}
