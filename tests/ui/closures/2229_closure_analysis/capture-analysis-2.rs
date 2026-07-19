//@ edition:2021

#![feature(rustc_attrs)]

#[derive(Debug)]
struct Point {
    x: String,
    y: i32,
}

fn main() {
    let mut p = Point { x: String::new(), y: 10 };

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR First Pass analysis includes:
    //~| ERROR Min Capture analysis includes:
        let _x = p.x;
        //~^ NOTE: Capturing p[(0, 0)] -> ByValue
        //~| NOTE: p[] captured as ByValue here
        println!("{:?}", p);
        //~^ NOTE: Capturing p[] -> Immutable
        //~| NOTE: Min Capture p[] -> ByValue
        //~| NOTE: p[] used here
    };
}
