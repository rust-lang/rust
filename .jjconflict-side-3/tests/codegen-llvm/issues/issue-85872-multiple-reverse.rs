//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

#[no_mangle]
pub fn u16_be_to_arch(mut data: [u8; 2]) -> [u8; 2] {
    // CHECK-LABEL: @u16_be_to_arch
    // CHECK: @llvm.bswap.i16
    data.reverse();
    data
}

#[no_mangle]
pub fn u32_be_to_arch(mut data: [u8; 4]) -> [u8; 4] {
    // CHECK-LABEL: @u32_be_to_arch
    // CHECK: @llvm.bswap.i32
    data.reverse();
    data
}
