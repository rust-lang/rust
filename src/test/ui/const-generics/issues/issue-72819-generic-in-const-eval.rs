// Regression test for #72819: ICE due to failure in resolving the const generic in `Arr`'s type
// bounds.
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct Arr<const N: usize>
where Assert::<{N < usize::max_value() / 2}>: IsTrue,
//[full]~^ ERROR constant expression depends on a generic parameter
//[min]~^^ ERROR generic parameters may not be used in const operations
{
}

enum Assert<const CHECK: bool> {}

trait IsTrue {}

impl IsTrue for Assert<true> {}

fn main() {
    let x: Arr<{usize::max_value()}> = Arr {};
}
