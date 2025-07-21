//@ add-core-stubs
//@ revisions: enable-backchain disable-backchain default-backchain
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --crate-type=lib --target=s390x-unknown-linux-gnu
//@ needs-llvm-components: systemz
//@[enable-backchain] compile-flags: -Ctarget-feature=+backchain
//@[disable-backchain] compile-flags: -Ctarget-feature=-backchain
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

extern "C" {
    fn extern_func();
}

// CHECK-LABEL: test_backchain
#[no_mangle]
extern "C" fn test_backchain() -> i32 {
    // Here we try to match if backchain register is saved to the parameter area (stored in r15/sp)
    // And also if a new parameter area (160 bytes) is allocated for the upcoming function call
    // enable-backchain: lgr [[REG1:.*]], %r15
    // enable-backchain-NEXT: aghi %r15, -160
    // enable-backchain: stg [[REG1]], 0(%r15)
    // disable-backchain: aghi %r15, -160
    // disable-backchain-NOT: stg %r{{.*}}, 0(%r15)
    // default-backchain: aghi %r15, -160
    // default-backchain-NOT: stg %r{{.*}}, 0(%r15)
    unsafe {
        extern_func();
    }
    // enable-backchain-NEXT: brasl %r{{.*}}, extern_func@PLT
    // disable-backchain: brasl %r{{.*}}, extern_func@PLT

    // Make sure that the expected return value is written into %r2 (return register):
    // enable-backchain-NEXT: lghi %r2, 1
    // disable-backchain: lghi %r2, 0
    // default-backchain: lghi %r2, 0
    #[cfg(target_feature = "backchain")]
    {
        1
    }
    #[cfg(not(target_feature = "backchain"))]
    {
        0
    }
    // CHECK: br %r{{.*}}
}
