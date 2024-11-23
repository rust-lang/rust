//@ run-pass
//@ reference: layout.aggregate.struct-size-align
//@ edition: 2018

#[allow(dead_code)]
union ReprRustUnion {
    x: i32,
    y: [u32; 4],
    z: f32,
    a: u128,
}

fn test_size_contains_each_type() {
    assert!(core::mem::size_of::<i32>() <= core::mem::size_of::<ReprRustUnion>());
    assert!(core::mem::size_of::<[u32; 4]>() <= core::mem::size_of::<ReprRustUnion>());
    assert!(core::mem::size_of::<f32>() <= core::mem::size_of::<ReprRustUnion>());
    assert!(core::mem::size_of::<u128>() <= core::mem::size_of::<ReprRustUnion>());
}

fn test_size_contains_all_fields() {
    assert!(
        (core::mem::offset_of!(ReprRustUnion, x) + core::mem::size_of::<i32>())
            <= core::mem::size_of::<ReprRustUnion>()
    );
    assert!(
        (core::mem::offset_of!(ReprRustUnion, y) + core::mem::size_of::<[u32; 4]>())
            <= core::mem::size_of::<ReprRustUnion>()
    );
    assert!(
        (core::mem::offset_of!(ReprRustUnion, z) + core::mem::size_of::<f32>())
            <= core::mem::size_of::<ReprRustUnion>()
    );
    assert!(
        (core::mem::offset_of!(ReprRustUnion, a) + core::mem::size_of::<u128>())
            <= core::mem::size_of::<ReprRustUnion>()
    );
}

fn test_size_modulo_align() {
    assert_eq!(core::mem::size_of::<ReprRustUnion>() % core::mem::align_of::<ReprRustUnion>(), 0);
}

fn main() {
    test_size_contains_each_type();
    test_size_contains_all_fields();
    test_size_modulo_align();
}
