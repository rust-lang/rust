//@ run-pass
//@ reference: layout.aggregate.struct-offsets
//@ edition: 2018

#[repr(align(64))]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub struct Overaligned(u8);

#[allow(dead_code)]
union ReprRustUnion {
    x: i32,
    y: [u32; 4],
    z: f32,
    a: u128,
    b: Overaligned,
}

fn test_fields_aligned() {
    assert_eq!((core::mem::offset_of!(ReprRustUnion, x) % (core::mem::align_of::<i32>())), 0);
    assert_eq!((core::mem::offset_of!(ReprRustUnion, y) % (core::mem::align_of::<[u32; 4]>())), 0);
    assert_eq!((core::mem::offset_of!(ReprRustUnion, z) % (core::mem::align_of::<f32>())), 0);
    assert_eq!((core::mem::offset_of!(ReprRustUnion, a) % (core::mem::align_of::<u128>())), 0);
    assert_eq!(
        (core::mem::offset_of!(ReprRustUnion, b) % (core::mem::align_of::<Overaligned>())),
        0
    );
}

fn main() {
    test_fields_aligned();
}
