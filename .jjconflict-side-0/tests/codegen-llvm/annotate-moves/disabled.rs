//@ compile-flags: -Zannotate-moves=no -Copt-level=0 -g
// Test that move/copy operations are NOT annotated when the flag is disabled

#![crate_type = "lib"]

struct LargeStruct {
    data: [u64; 20], // 160 bytes - would normally trigger annotation
}

impl Clone for LargeStruct {
    // CHECK-LABEL: <disabled::LargeStruct as core::clone::Clone>::clone
    fn clone(&self) -> Self {
        // Should NOT be annotated when flag is disabled
        // CHECK-NOT: compiler_copy
        LargeStruct { data: self.data }
    }
}

// CHECK-LABEL: disabled::test_large_copy_no_annotation
pub fn test_large_copy_no_annotation() {
    let large = LargeStruct { data: [42; 20] };
    // CHECK-NOT: compiler_copy
    let _copy = large.clone();
}

// CHECK-LABEL: disabled::test_large_move_no_annotation
pub fn test_large_move_no_annotation() {
    let large = LargeStruct { data: [42; 20] };
    // CHECK-NOT: compiler_move
    let _moved = large;
}

// Verify that no compiler_move or compiler_copy annotations exist anywhere
// CHECK-NOT: compiler_move
// CHECK-NOT: compiler_copy
