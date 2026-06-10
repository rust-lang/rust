//@ compile-flags: -C opt-level=3 --target powerpc64le-unknown-linux-gnu
//@ needs-llvm-components: powerpc

#![crate_type = "lib"]
#![feature(stdarch_powerpc)]

use std::arch::powerpc64::*;

// CHECK-LABEL: @test_vec_add_double
#[no_mangle]
pub unsafe fn test_vec_add_double(a: vector_double, b: vector_double) -> vector_double {
    // CHECK: fadd <2 x double>
    vec_add(a, b)
}

// Made with Bob
