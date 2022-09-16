// This previously compiled, but was intentionally changed in #101478.
// This was used in https://github.com/GoldsteinE/name-it.
//
// See that PR for more details.
//
// check-pass
use std::mem;
fn returns_opaque() -> impl Sized {
    0u8
}

struct NamedOpaqueType {
    data: [mem::MaybeUninit<u8>; size_of_fut(returns_opaque)]
    //~^ WARNING relying on the underlying type of an opaque type in the type system
    //~| WARNING this was previously accepted by the compiler
}

const fn size_of_fut<FUT>(x: fn() -> FUT) -> usize {
   mem::size_of::<FUT>()
}

fn main() {}
