//@ add-minicore
//@ revisions: disable-softfloat enable-softfloat
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --crate-type=lib

// we expect the build to fail in the feature
//@ build-pass
//@ [enable-softfloat] compile-flags: --target=s390x-unknown-none-softfloat
//@ [enable-softfloat] -C target-feature=+vector
//@ [enable-softfloat] needs-llvm-components: systemz
//@ [disable-softfloat] compile-flags: --target=s390x-unknown-linux-gnu
//@ [disable-softfloat] -C target-feature=+soft-float
//@ [disable-softfloat] needs-llvm-components: systemz

//[disable-softfloat]~? WARN target feature `soft-float` must be disabled to ensure that the ABI of the current target can be implemented correctly
//[disable-softfloat]~? WARN target feature `soft-float` cannot be enabled with `-Ctarget-feature`
//[enable-softfloat]~? WARN target feature `vector` must be disabled to ensure that the ABI of the current target can be implemented correctly

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

extern "C" {
    fn extern_func(value: f64) -> f64;
}

#[no_mangle]
extern "C" fn test_softfloat() -> f64 {
    let value = 3.141_f64;

    unsafe { extern_func(value) } ;

    2.718_f64
}
