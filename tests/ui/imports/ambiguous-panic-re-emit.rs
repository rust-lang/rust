#![crate_type = "lib"]
#![no_std]

macro_rules! re_emit {
    ($($i:item)*) => ($($i)*)
}

// By re-emitting the prelude import via a macro, we run into the delayed bugs code path.
re_emit! {
    extern crate std;
    use std::prelude::v1::*;
}

fn xx() {
    panic!();
    //~^ WARNING `panic` is ambiguous
    //~| WARNING this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!

    // We can't deny the above lint, or else it *won't* run into the problematic issue of *not*
    // having reported an error. So we crate a dummy error.
    let _ = unknown_item;
    //~^ ERROR: cannot find value `unknown_item`
}
