// Regression test for #72819: ICE due to failure in resolving the const generic in `Arr`'s type
// bounds.

#![feature(const_generics)]
#![allow(incomplete_features)]
struct Arr<const N: usize>
where Assert::<{N < usize::max_value() / 2}>: IsTrue,
//~^ ERROR constant expression depends on a generic parameter
{
}

enum Assert<const CHECK: bool> {}

trait IsTrue {}

impl IsTrue for Assert<true> {}

fn main() {
    let x: Arr<{usize::max_value()}> = Arr {};
}
