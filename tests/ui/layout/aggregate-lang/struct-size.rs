//@ run-pass
//@ reference: layout.aggregate.struct-size-align
//@ edition: 2018

#[allow(dead_code)]
struct ReprRustStruct {
    x: i32,
    y: [u32; 4],
    z: f32,
    a: u128,
}

fn test_size_contains_all_types() {
    assert!(
        core::mem::size_of::<ReprRustStruct>()
            >= (core::mem::size_of::<i32>()
                + core::mem::size_of::<[u32; 4]>()
                + core::mem::size_of::<f32>()
                + core::mem::size_of::<u128>())
    );
}

fn test_size_contains_all_fields() {
    assert!(
        (core::mem::offset_of!(ReprRustStruct, x) + core::mem::size_of::<i32>())
            <= core::mem::size_of::<ReprRustStruct>()
    );
    assert!(
        (core::mem::offset_of!(ReprRustStruct, y) + core::mem::size_of::<[u32; 4]>())
            <= core::mem::size_of::<ReprRustStruct>()
    );
    assert!(
        (core::mem::offset_of!(ReprRustStruct, z) + core::mem::size_of::<f32>())
            <= core::mem::size_of::<ReprRustStruct>()
    );
    assert!(
        (core::mem::offset_of!(ReprRustStruct, a) + core::mem::size_of::<u128>())
            <= core::mem::size_of::<ReprRustStruct>()
    );
}

fn test_size_modulo_align() {
    assert_eq!(core::mem::size_of::<ReprRustStruct>() % core::mem::align_of::<ReprRustStruct>(), 0);
}

fn main() {
    test_size_contains_all_fields();
    test_size_contains_all_types();
    test_size_modulo_align();
}
