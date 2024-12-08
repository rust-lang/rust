//@ edition:2021

#![feature(rustc_attrs)]

fn main() {
    let s = format!("s");

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("This uses new capture analyysis to capture s={}", s);
        //~^ NOTE: Capturing s[] -> Immutable
        //~| NOTE: Min Capture s[] -> Immutable
    };
}
