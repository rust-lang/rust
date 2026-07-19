//@ edition:2021

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
    || {
    //~^ ERROR First Pass analysis includes:
    //~| ERROR Min Capture analysis includes:
        println!("{:?}", p);
        //~^ NOTE: Capturing p[] -> Immutable
        //~| NOTE: Min Capture p[] -> Immutable
        println!("{:?}", p.x);
        //~^ NOTE: Capturing p[(0, 0)] -> Immutable

        println!("{:?}", q.x);
        //~^ NOTE: Capturing q[(0, 0)] -> Immutable
        println!("{:?}", q);
        //~^ NOTE: Capturing q[] -> Immutable
        //~| NOTE: Min Capture q[] -> Immutable
    };
}
