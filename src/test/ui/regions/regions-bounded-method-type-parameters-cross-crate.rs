// aux-build:rbmtp_cross_crate_lib.rs

// Check explicit region bounds on methods in the cross crate case.

extern crate rbmtp_cross_crate_lib as lib;

use lib::Inv;
use lib::MaybeOwned;
use lib::IntoMaybeOwned;

fn call_into_maybe_owned<'x,F:IntoMaybeOwned<'x>>(f: F) {
    // Exercise a code path I found to be buggy. We were not encoding
    // the region parameters from the receiver correctly on trait
    // methods.
    f.into_maybe_owned();
}

fn call_bigger_region<'x, 'y>(a: Inv<'x>, b: Inv<'y>) {
    // Here the value provided for 'y is 'y, and hence 'y:'x does not hold.
    a.bigger_region(b) //~ ERROR lifetime mismatch [E0623]
}

fn main() { }
