// run-pass
#![allow(unused_variables)]
// compile-flags: --extern LooksLikeExternCrate

mod m {
    pub struct LooksLikeExternCrate;
}

fn main() {
    // OK, speculative resolution for `unused_qualifications` doesn't try
    // to resolve this as an extern crate and load that crate
    let s = m::LooksLikeExternCrate {};
}
