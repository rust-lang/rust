// edition:2021
#![feature(rustc_attrs)]

// Ensure that capture analysis results in arrays being completely captured.
fn main() {
    let mut m = [1, 2, 3, 4, 5];

    let mut c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        m[0] += 10;
        //~^ NOTE: Capturing m[] -> MutBorrow
        //~| NOTE: Min Capture m[] -> MutBorrow
        m[1] += 40;
        //~^ NOTE: Capturing m[] -> MutBorrow
    };

    c();
}
