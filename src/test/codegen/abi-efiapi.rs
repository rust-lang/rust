// Checks if the correct annotation for the efiapi ABI is passed to llvm.

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(abi_efiapi)]

// CHECK: define win64 i64 @has_efiapi
#[no_mangle]
#[cfg(target_arch = "x86_64")]
pub extern "efiapi" fn has_efiapi(a: i64) -> i64 {
    a * 2
}

// CHECK: define c i64 @has_efiapi
#[no_mangle]
#[cfg(not(target_arch = "x86_64"))]
pub extern "efiapi" fn has_efiapi(a: i64) -> i64 {
    a * 2
}
