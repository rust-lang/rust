//
//@ compile-flags: -Z annotate-moves=100 -Copt-level=0 -g

#![crate_type = "lib"]

#[derive(Clone)]
struct MediumStruct {
    data: [u64; 10], // 80 bytes - below custom 100-byte threshold
}

#[derive(Clone)]
struct LargeStruct {
    data: [u64; 20], // 160 bytes - above custom 100-byte threshold
}

impl MediumStruct {
    fn new() -> Self {
        MediumStruct { data: [42; 10] }
    }
}

impl LargeStruct {
    fn new() -> Self {
        LargeStruct { data: [42; 20] }
    }
}

// With custom size limit of 100 bytes:
// Medium struct (80 bytes) should NOT be annotated
// Large struct (160 bytes) should be annotated

// CHECK-LABEL: annotate_moves_size_limit::test_medium_copy
// CHECK-NOT: !DISubprogram(name: "compiler_copy"
pub fn test_medium_copy() {
    let medium = MediumStruct::new();
    let _copy = medium.clone(); // Should NOT generate annotation (below threshold)
}

// CHECK-LABEL: annotate_moves_size_limit::test_large_copy
pub fn test_large_copy() {
    let large = LargeStruct::new();
    let _copy = large.clone(); // Should generate annotation (above threshold)
}

// Check that compiler_copy debug info is generated for large copies
// (above 100-byte threshold) with size parameter
// CHECK: !DISubprogram(name: "compiler_copy<[u64; 20], 160>"
