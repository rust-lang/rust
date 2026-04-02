//@ check-pass

#![feature(const_attr_paths)]

mod foo {
    pub const fn align() -> usize {
        8
    }

    pub const ALIGN: usize = align();
}

const COMPUTED_ALIGN: usize = core::mem::align_of::<u32>();
const PACK: usize = 2;
use foo::ALIGN as IMPORTED_ALIGN;

#[repr(align(foo::ALIGN))]
struct Aligned(u8);

#[repr(align(IMPORTED_ALIGN))]
struct Imported(u8);

#[repr(align(COMPUTED_ALIGN))]
struct Computed(u8);

#[repr(packed(PACK))]
struct Packed(u32);

fn main() {
    let _ = Aligned(0);
    let _ = Imported(0);
    let _ = Computed(0);
    let _ = Packed(0);
}
