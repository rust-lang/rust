//@ add-minicore
//@ needs-llvm-components: aarch64
//@ compile-flags: --target=arm64e-apple-ios -Copt-level=0 -Cpanic=abort

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

extern "C" {
    fn external_c();
}

// CHECK-DAG: @CUSTOM_FINI = {{.*}}ptr ptrauth (ptr @external_c, i32 0), section ".fini_array,__data"
#[used]
#[no_mangle]
#[link_section = ".fini_array,__data"]
pub static CUSTOM_FINI: unsafe extern "C" fn() = external_c;

// CHECK-DAG: @CUSTOM_INIT = {{.*}}ptr ptrauth (ptr @external_c, i32 0), section ".init_array,__data"
#[used]
#[no_mangle]
#[link_section = ".init_array,__data"]
pub static CUSTOM_INIT: unsafe extern "C" fn() = external_c;
