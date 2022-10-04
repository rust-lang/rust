// This pattern is used in https://github.com/GoldsteinE/name-it and requires us
// to reveal opaque types during ctfe, even with `Reveal::UserFacing`.
//
// check-pass
use std::mem;
fn returns_opaque() -> impl Sized {
    0u8
}

struct Wrapper<T>(T);
fn returns_wrapped() -> impl Sized {
    Wrapper(returns_opaque())
}

struct NamedOpaqueType {
    data: [mem::MaybeUninit<u8>; size_of_fut(returns_opaque)]
}

struct NamedOpaqueTypeWrapper {
    data: [mem::MaybeUninit<u8>; size_of_fut(returns_wrapped)]
}

const fn size_of_fut<FUT>(x: fn() -> FUT) -> usize {
   mem::size_of::<FUT>()
}

fn main() {}
