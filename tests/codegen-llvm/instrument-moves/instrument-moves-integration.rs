//@ compile-flags: -Z instrument-moves -Z instrument-moves-size-limit=50 -Copt-level=0 -g

#![crate_type = "lib"]

// Comprehensive integration test for move/copy instrumentation

#[derive(Clone)]
struct VerySmall {
    x: u8, // 1 byte - should not be instrumented
}

#[derive(Clone)]
struct Small {
    data: [u32; 10], // 40 bytes - below 50-byte threshold
}

#[derive(Clone)]
struct Medium {
    data: [u64; 10], // 80 bytes - above 50-byte threshold
}

#[derive(Clone)]
struct Large {
    data: [u64; 30], // 240 bytes - well above threshold
}

// Test 1: Very small types should never be instrumented
// CHECK-LABEL: instrument_moves_integration::test_very_small_operations
// CHECK-NOT: !DISubprogram(name: "compiler_copy"
// CHECK-NOT: !DISubprogram(name: "compiler_move"
pub fn test_very_small_operations() {
    let vs = VerySmall { x: 42 };
    let _copy = vs.clone();
    let _moved = vs;
}

// Test 2: Small types below threshold should not be instrumented
// CHECK-LABEL: instrument_moves_integration::test_small_operations
// CHECK-NOT: !DISubprogram(name: "compiler_copy"
// CHECK-NOT: !DISubprogram(name: "compiler_move"
pub fn test_small_operations() {
    let s = Small { data: [42; 10] };
    let _copy = s.clone();
    let _moved = s;
}

// Test 3: Medium types above threshold should be instrumented
// CHECK-LABEL: instrument_moves_integration::test_medium_copy
pub fn test_medium_copy() {
    let m = Medium { data: [42; 10] };
    let _copy = m.clone(); // Should be instrumented
}

// CHECK-LABEL: instrument_moves_integration::test_medium_move
pub fn test_medium_move() {
    let m = Medium { data: [42; 10] };
    let _moved = m; // Should be instrumented
}

// Test 4: Large types should definitely be instrumented
// CHECK-LABEL: instrument_moves_integration::test_large_copy
pub fn test_large_copy() {
    let l = Large { data: [42; 30] };
    let _copy = l.clone(); // Should be instrumented
}

// CHECK-LABEL: instrument_moves_integration::test_large_move
pub fn test_large_move() {
    let l = Large { data: [42; 30] };
    let _moved = l; // Should be instrumented
}

// Test 5: Multiple operations in same function
// CHECK-LABEL: instrument_moves_integration::test_multiple_operations
pub fn test_multiple_operations() {
    let l1 = Large { data: [1; 30] };
    let _copy1 = l1.clone(); // Should be instrumented
    let _moved1 = l1; // Should be instrumented

    let l2 = Large { data: [2; 30] };
    let _copy2 = l2.clone(); // Should be instrumented
    drop(_copy2);
}

// Test 6: Function parameters and returns
// CHECK-LABEL: instrument_moves_integration::test_function_parameters
pub fn test_function_parameters() {
    let l = Large { data: [42; 30] };
    helper_function(l); // Should be instrumented for move into function
}

// CHECK-LABEL: instrument_moves_integration::helper_function
pub fn helper_function(_param: Large) {
    // Parameter receipt shouldn't be instrumented (it's the caller's move)
}

// Test 7: Verify ZST types are never instrumented
pub struct ZeroSizedType;

// CHECK-LABEL: instrument_moves_integration::test_zst_operations
// CHECK-NOT: !DISubprogram(name: "compiler_copy"
// CHECK-NOT: !DISubprogram(name: "compiler_move"
pub fn test_zst_operations() {
    let zst = ZeroSizedType;
    let _copy = zst;
    let _moved = ZeroSizedType;
}

// Check that compiler debug info is generated for medium and large types
// (above 50-byte threshold) with size parameter
// CHECK: !DISubprogram(name: "compiler_copy<[u64; 10], 80>"
// CHECK: !DISubprogram(name: "compiler_move<[u64; 10], 80>"
// CHECK: !DISubprogram(name: "compiler_copy<[u64; 30], 240>"
// CHECK: !DISubprogram(name: "compiler_move<[u64; 30], 240>"
