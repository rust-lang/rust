#![feature(pointer_sized_trait)]

use std::marker::PointerSized;

fn require_pointer_sized(_: impl PointerSized) {}

fn main() {
    require_pointer_sized(1usize);
    require_pointer_sized(1u16);
    //~^ ERROR `u16` needs to be a pointer-sized type
    require_pointer_sized(&1i16);
}
