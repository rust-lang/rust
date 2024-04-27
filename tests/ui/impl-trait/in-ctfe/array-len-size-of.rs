//! Check that const eval can use the size of opaque types.
//@ check-pass
use std::mem;
fn returns_opaque() -> impl Sized {
    0u8
}

struct NamedOpaqueType {
    data: [mem::MaybeUninit<u8>; size_of_fut(returns_opaque)],
}

const fn size_of_fut<FUT>(x: fn() -> FUT) -> usize {
    mem::size_of::<FUT>()
}

fn main() {}
