//@ add-minicore
//@ compile-flags: -C opt-level=3 --target powerpc64le-unknown-linux-gnu
//@ needs-llvm-components: powerpc

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

use core::arch::powerpc64::*;

// CHECK-LABEL: @test_vec_add_double
#[no_mangle]
pub unsafe fn test_vec_add_double(a: vector_double, b: vector_double) -> vector_double {
    // CHECK: fadd <2 x double>
    vec_add(a, b)
}
