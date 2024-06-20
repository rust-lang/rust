// Ensure the asm for array comparisons is properly optimized.

//@ compile-flags: -C opt-level=2

#![crate_type = "lib"]

// CHECK-LABEL: @compare
// CHECK: start:
// CHECK-NEXT: ret i1 true
#[no_mangle]
pub fn compare() -> bool {
    let bytes = 12.5f32.to_ne_bytes();
    bytes
        == if cfg!(target_endian = "big") {
            [0x41, 0x48, 0x00, 0x00]
        } else {
            [0x00, 0x00, 0x48, 0x41]
        }
}
