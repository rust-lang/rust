//
//@ compile-flags: -Copt-level=0 -g

// Test that move/copy operations are NOT annotated when the flag is disabled

#![crate_type = "lib"]

#[derive(Clone)]
struct LargeStruct {
    data: [u64; 20], // 160 bytes - would normally trigger annotation
}

impl LargeStruct {
    fn new() -> Self {
        LargeStruct { data: [42; 20] }
    }
}

// Without -Z annotate-moves flag, no annotation should be generated

// CHECK-LABEL: annotate_moves_disabled::test_large_copy_no_annotation
// CHECK-NOT: !DISubprogram(name: "compiler_copy"
pub fn test_large_copy_no_annotation() {
    let large = LargeStruct::new();
    let _copy = large.clone();
}

// CHECK-LABEL: annotate_moves_disabled::test_large_move_no_annotation
// CHECK-NOT: !DISubprogram(name: "compiler_move"
pub fn test_large_move_no_annotation() {
    let large = LargeStruct::new();
    let _moved = large;
}
