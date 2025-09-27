//
//@ compile-flags: -Copt-level=0 -g

#![crate_type = "lib"]

#[derive(Clone)]
struct LargeStruct {
    data: [u64; 50], // 400 bytes - would normally trigger instrumentation if enabled
}

impl LargeStruct {
    fn new() -> Self {
        LargeStruct { data: [42; 50] }
    }
}

// Without -Z instrument-moves flag, no instrumentation should be generated

// CHECK-LABEL: instrument_moves_disabled::test_large_copy_no_instrumentation
// CHECK-NOT: !DISubprogram(name: "compiler_copy"
pub fn test_large_copy_no_instrumentation() {
    let large = LargeStruct::new();
    let _copy = large.clone(); // Should NOT generate instrumentation (feature disabled)
}

// CHECK-LABEL: instrument_moves_disabled::test_large_move_no_instrumentation
// CHECK-NOT: !DISubprogram(name: "compiler_move"
pub fn test_large_move_no_instrumentation() {
    let large = LargeStruct::new();
    let _moved = large; // Should NOT generate instrumentation (feature disabled)
}
