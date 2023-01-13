#![feature(rustc_attrs)]

// edition:2021

// Test that any precise capture on a union is truncated because it's unsafe to do so.

union Union {
    value: u64,
}

fn main() {
    let u = Union { value: 42 };

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
       unsafe { u.value }
        //~^ NOTE: Capturing u[(0, 0)] -> ImmBorrow
        //~| NOTE: Min Capture u[] -> ImmBorrow
    };

    c();
}
