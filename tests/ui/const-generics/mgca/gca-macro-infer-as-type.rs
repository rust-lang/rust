//! `gca!(_)` used to be allowed to infer to a type as a compiler implementation quirk. Since it
//! uses explicit const argument syntax, it is now rejected when passed as a type argument
#![feature(min_generic_const_args)]

use std::gca;

struct S<T>(T);

fn main() {
    let _: S<gca!(_)> = S(2u32);
    //~^ ERROR: constant provided when a type was expected
    let _: S<{ gca!(_) }> = S(2u32);
    //~^ ERROR: constant provided when a type was expected
}
