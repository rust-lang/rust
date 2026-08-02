// Checks that 32-bit SPARC passes and returns `f128` indirectly, like C `long double`. GCC and
// Clang both use an `sret` pointer for the return value, which is also what makes LLVM emit the
// `unimp` marker after the call and the matching `%o7+12` return in the callee.

//@ add-minicore
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes --target=sparc-unknown-linux-gnu
//@ needs-llvm-components: sparc

#![feature(no_core, lang_items, f128)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK-LABEL: define void @ret_f128(ptr {{.*}}sret([16 x i8]) align 8 {{.*}})
#[no_mangle]
pub extern "C" fn ret_f128() -> f128 {
    1.5
}

// CHECK-LABEL: define void @id_f128(ptr {{.*}}sret([16 x i8]) align 8 {{.*}}, ptr {{.*}}byval([16 x i8]) align 8 {{.*}})
#[no_mangle]
pub extern "C" fn id_f128(x: f128) -> f128 {
    x
}

extern "C" {
    fn take_f128(x: f128);
}

// CHECK-LABEL: define void @call_f128(
// CHECK: call void @take_f128(ptr {{.*}}byval([16 x i8]) align 8 {{.*}})
#[no_mangle]
pub extern "C" fn call_f128(x: f128) {
    unsafe { take_f128(x) }
}

// `f32` and `f64` are still passed and returned directly.
// CHECK-LABEL: define{{.*}} float @id_f32(float {{.*}})
#[no_mangle]
pub extern "C" fn id_f32(x: f32) -> f32 {
    x
}

// CHECK-LABEL: define{{.*}} double @id_f64(double {{.*}})
#[no_mangle]
pub extern "C" fn id_f64(x: f64) -> f64 {
    x
}
