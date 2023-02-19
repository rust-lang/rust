// compile-flags: -C opt-level=3 -C target-cpu=x86-64-v3
// no-system-llvm
// only-x86_64
// ignore-debug (the extra assertions get in the way)

#![crate_type = "lib"]
#![feature(array_zip)]

// CHECK-LABEL: @short_integer_map
#[no_mangle]
pub fn short_integer_map(x: [u32; 8]) -> [u32; 8] {
    // CHECK: load <8 x i32>
    // CHECK: shl <8 x i32>
    // CHECK: or <8 x i32>
    // CHECK: store <8 x i32>
    x.map(|x| 2 * x + 1)
}

// CHECK-LABEL: @short_integer_zip_map
#[no_mangle]
pub fn short_integer_zip_map(x: [u32; 8], y: [u32; 8]) -> [u32; 8] {
    // CHECK: %[[A:.+]] = load <8 x i32>
    // CHECK: %[[B:.+]] = load <8 x i32>
    // CHECK: sub <8 x i32> %[[A]], %[[B]]
    // CHECK: store <8 x i32>
    x.zip(y).map(|(x, y)| x - y)
}

// This test is checking that LLVM can SRoA away a bunch of the overhead,
// like fully moving the iterators to registers.  Notably, previous implementations
// of `map` ended up `alloca`ing the whole `array::IntoIterator`, meaning both a
// hard-to-eliminate `memcpy` and that the iteration counts needed to be written
// out to stack every iteration, even for infallible operations on `Copy` types.
//
// This is still imperfect, as there's more copies than would be ideal,
// but hopefully work like #103830 will improve that in future,
// and update this test to be stricter.
//
// CHECK-LABEL: @long_integer_map
#[no_mangle]
pub fn long_integer_map(x: [u32; 64]) -> [u32; 64] {
    // CHECK: start:
    // CHECK-NEXT: alloca [64 x i32]
    // CHECK-NEXT: alloca %"core::mem::manually_drop::ManuallyDrop<[u32; 64]>"
    // CHECK-NOT: alloca
    // CHECK: mul <{{[0-9]+}} x i32>
    // CHECK: add <{{[0-9]+}} x i32>
    x.map(|x| 13 * x + 7)
}
