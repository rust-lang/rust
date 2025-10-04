//
//@ compile-flags: -Z instrument-moves -Copt-level=0 -g

#![crate_type = "lib"]

#[derive(Clone)]
struct LargeStruct {
    data: [u64; 20], // 160 bytes - should trigger instrumentation (above 64 bytes)
}

#[derive(Clone)]
struct SmallStruct {
    data: u32, // 4 bytes - should NOT trigger instrumentation
}

impl LargeStruct {
    fn new() -> Self {
        LargeStruct { data: [42; 20] }
    }
}

impl SmallStruct {
    fn new() -> Self {
        SmallStruct { data: 42 }
    }
}

// CHECK-LABEL: instrument_moves::test_large_copy
pub fn test_large_copy() {
    let large = LargeStruct::new();
    let _copy = large.clone(); // Should generate instrumentation debug info
}

// CHECK-LABEL: instrument_moves::test_large_move
pub fn test_large_move() {
    let large = LargeStruct::new();
    let _moved = large; // Should generate instrumentation debug info
}

// CHECK-LABEL: instrument_moves::test_small_copy
// CHECK-NOT: !DISubprogram(name: "compiler_copy"
pub fn test_small_copy() {
    let small = SmallStruct::new();
    let _copy = small.clone(); // Should NOT generate instrumentation debug info
}

// CHECK-LABEL: instrument_moves::test_small_move
// CHECK-NOT: !DISubprogram(name: "compiler_move"
pub fn test_small_move() {
    let small = SmallStruct::new();
    let _moved = small; // Should NOT generate instrumentation debug info
}

// Check that compiler_copy debug info is generated for large copies with size parameter
// CHECK: !DISubprogram(name: "compiler_copy<[u64; 20], 160>"

// Check that compiler_move debug info is generated for large moves with size parameter
// CHECK: !DISubprogram(name: "compiler_move<[u64; 20], 160>"
