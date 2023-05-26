// check-pass
// compile-flags: -Ztrait-solver=next
#![feature(specialization)]
#![allow(incomplete_features)]
trait Trait {
    type Assoc;
}

impl<T> Trait for T {
    default type Assoc = u32;
}

impl Trait for u32 {
    type Assoc = u32;
}

fn generic<T: Trait<Assoc = u32>>(_: T) {}

fn main() {
    generic(1)
    // We want `normalizes-to(<{integer} as Trait>::Assoc, u32)`
    // to succeed as there is only one impl that can be used for
    // this function to compile, even if the default impl would
    // also satisfy this. This is different from coherence where
    // doing so would be unsound.
}
