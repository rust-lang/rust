//@ edition:2021

#![feature(rustc_attrs)]

fn main() {
    let s = format!("s");

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("This uses new capture analyysis to capture s={}", s);
        //~^ NOTE: Capturing s[] -> Immutable
        //~| NOTE: Min Capture s[] -> Immutable
    };
}
