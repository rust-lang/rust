#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
struct Arr<const N: usize>
where
    Assert::<{N < usize::MAX / 2}>: IsTrue,
{}

enum Assert<const CHECK: bool> {}

trait IsTrue {}

impl IsTrue for Assert<true> {}

fn main() {
    let x: Arr<{usize::MAX}> = Arr {};
    //~^ ERROR mismatched types
    //~| ERROR mismatched types
}
