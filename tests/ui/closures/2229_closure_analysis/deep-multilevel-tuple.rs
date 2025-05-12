//@ edition:2021
#![feature(rustc_attrs)]
#![allow(unused)]

fn main() {
    let mut t = (((1,2),(3,4)),((5,6),(7,8)));

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        let x = &t.0.0.0;
        //~^ NOTE: Capturing t[(0, 0),(0, 0),(0, 0)] -> Immutable
        t.1.1.1 = 9;
        //~^ NOTE: Capturing t[(1, 0),(1, 0),(1, 0)] -> Mutable
        //~| NOTE: t[] captured as Mutable here
        println!("{:?}", t);
        //~^ NOTE: Min Capture t[] -> Mutable
        //~| NOTE: Capturing t[] -> Immutable
        //~| NOTE: t[] used here
    };
}
