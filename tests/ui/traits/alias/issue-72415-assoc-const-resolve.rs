//@ check-pass

#![feature(trait_alias)]

trait Bounded { const MAX: Self; }

impl Bounded for u32 {
    // This should correctly resolve to the associated const in the inherent impl of u32.
    const MAX: Self = u32::MAX;
}

trait Num = Bounded + Copy;

fn main() {}
