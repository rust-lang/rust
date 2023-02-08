// compile-flags: -Ztrait-solver=next

#![feature(pointer_like_trait)]

use std::marker::PointerLike;

fn require_(_: impl PointerLike) {}

fn main() {
    require_(1usize);
    require_(1u16);
    //~^ ERROR `u16` needs to have the same alignment and size as a pointer
    require_(&1i16);
}
