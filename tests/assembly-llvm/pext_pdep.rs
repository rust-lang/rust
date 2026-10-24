//@ assembly-output: emit-asm
//@ compile-flags: -C opt-level=3 -C target-feature=+bmi2
//@ only-x86_64-unknown-linux-gnu
//@ min-llvm-version: 23

#![crate_type = "lib"]
#![feature(uint_gather_scatter_bits)]

// CHECK-LABEL: test_pext_u32:
// CHECK: pextl
#[no_mangle]
pub fn test_pext_u32(a: u32, mask: u32) -> u32 {
    a.extract_bits(mask)
}

// CHECK-LABEL: test_pdep_u32:
// CHECK: pdepl
#[no_mangle]
pub fn test_pdep_u32(a: u32, mask: u32) -> u32 {
    a.deposit_bits(mask)
}

// CHECK-LABEL: test_pext_u64:
// CHECK: pextq
#[no_mangle]
pub fn test_pext_u64(a: u64, mask: u64) -> u64 {
    a.extract_bits(mask)
}

// CHECK-LABEL: test_pdep_u64:
// CHECK: pdepq
#[no_mangle]
pub fn test_pdep_u64(a: u64, mask: u64) -> u64 {
    a.deposit_bits(mask)
}
