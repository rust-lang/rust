//@ revisions: classic partial_init
//@[partial_init] check-pass

// Test that we don't allow partial initialization.
// This may be relaxed in the future (see #54987).

#![cfg_attr(partial_init, feature(partial_init_locals))]

fn main() {
    let mut t: (u64, u64);
    t.0 = 1;
    //[classic]~^ ERROR E0381
    //[classic]~| ERROR E0658
    t.1 = 1;

    let mut t: (u64, u64);
    t.1 = 1;
    //[classic]~^ ERROR E0381
    //[classic]~| ERROR E0658
    t.0 = 1;

    let mut t: (u64, u64);
    t.0 = 1;
    //[classic]~^ ERROR E0381
    //[classic]~| ERROR E0658

    let mut t: (u64,);
    t.0 = 1;
    //[classic]~^ ERROR E0381
    //[classic]~| ERROR E0658
}
