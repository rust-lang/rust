//@ edition:2021

#![feature(rustc_attrs)]

fn main() {
    let mut t = (10, 10);

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
    || {
    //~^ ERROR First Pass analysis includes:
    //~| ERROR Min Capture analysis includes:
        println!("{}", t.0);
        //~^ NOTE: Capturing t[(0, 0)] -> Immutable
        //~| NOTE: Min Capture t[(0, 0)] -> Immutable
    };

    // `c` only captures t.0, therefore mutating t.1 is allowed.
    let t1 = &mut t.1;

    c();
    *t1 = 20;
}
