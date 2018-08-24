// Checks if the correct annotation for the sysv64 ABI is passed to
// llvm. Also checks that the abi-sysv64 feature gate allows usage
// of the sysv64 abi.

// ignore-arm
// ignore-aarch64

// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]

// CHECK: define x86_64_sysvcc i64 @has_sysv64_abi
#[no_mangle]
pub extern "sysv64" fn has_sysv64_abi(a: i64) -> i64 {
    a * 2
}
