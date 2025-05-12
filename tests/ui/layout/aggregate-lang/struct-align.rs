//@ run-pass
//@ reference: layout.aggregate.struct-size-align
//@ edition: 2018

#[repr(align(64))]
#[derive(Copy, Clone)]
#[allow(dead_code)]
pub struct Overaligned(u8);

#[allow(dead_code)]
struct ReprRustStruct {
    x: i32,
    y: [u32; 4],
    z: f32,
    a: u128,
    b: Overaligned,
}

fn test_alignment_contains_all_fields() {
    assert!(core::mem::align_of::<ReprRustStruct>() >= core::mem::align_of::<i32>());
    assert!(core::mem::align_of::<ReprRustStruct>() >= core::mem::align_of::<[u32; 4]>());
    assert!(core::mem::align_of::<ReprRustStruct>() >= core::mem::align_of::<f32>());
    assert!(core::mem::align_of::<ReprRustStruct>() >= core::mem::align_of::<u128>());
    assert!(core::mem::align_of::<ReprRustStruct>() >= core::mem::align_of::<Overaligned>());
}

fn main() {
    test_alignment_contains_all_fields();
}
