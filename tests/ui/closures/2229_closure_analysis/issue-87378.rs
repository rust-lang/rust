#![feature(rustc_attrs)]

//@ edition:2021

// Test that any precise capture on a union is truncated because it's unsafe to do so.

union Union {
    value: u64,
}

fn main() {
    let u = Union { value: 42 };

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
       unsafe { u.value }
        //~^ NOTE: Capturing u[(0, 0)] -> Immutable
        //~| NOTE: Min Capture u[] -> Immutable
    };

    c();
}
