// This previously compiled, but was intentionally changed in #101478.
// This was used in https://github.com/GoldsteinE/name-it.
//
// See that PR for more details.
use std::mem;
fn returns_opaque() -> impl Sized {
    0u8
}

struct NamedOpaqueType {
    data: [mem::MaybeUninit<u8>; size_of_fut(returns_opaque)]
    //~^ ERROR unable to use constant with a hidden value in the type system
}

const fn size_of_fut<FUT>(x: fn() -> FUT) -> usize {
   mem::size_of::<FUT>()
}

fn main() {}
