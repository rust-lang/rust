// Check that LLVM can see that stores to a by-move array (passed indirectly) are dead.

// compile-flags: -O -Zmir-opt-level=0
// min-llvm-version: 14.0

#![crate_type="lib"]

// CHECK-LABEL: @array_dead_store
#[no_mangle]
pub fn array_dead_store(mut x: [u8; 1234]) {
    // CHECK-NOT: store
    x[10] = 42;
}
