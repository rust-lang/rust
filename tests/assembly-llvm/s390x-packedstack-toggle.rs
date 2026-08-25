//@ add-minicore
//@ revisions: enable-packedstack default-packedstack
//@ assembly-output: emit-asm
//@ compile-flags: -Copt-level=3 --crate-type=lib --target=s390x-unknown-linux-gnu -Cforce-unwind-tables=no
//@ needs-llvm-components: systemz
//@[enable-packedstack] compile-flags: -Zpacked-stack
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

extern "C" {
    fn extern_func() -> i32;
}

// CHECK-LABEL: test_packedstack
#[no_mangle]
extern "C" fn test_packedstack() -> i32 {
    // test the creation of call stack with and without packed-stack

    // without packed-stack we always reserve a least the maximal space of 160 bytes
    // default-packedstack: stmg %r14, %r15, 112(%r15)
    // default-packedstack-NEXT: aghi %r15, -160
    // default-packedstack-NEXT: brasl %r14, extern_func

    // with packed-stack only the actually needed registers are reserved on the stack
    // enable-packedstack: stmg %r14, %r15, 144(%r15)
    // enable-packedstack-NEXT: aghi %r15, -16
    // enable-packedstack-NEXT: brasl %r14, extern_func
    unsafe {
        extern_func();
    }
    1
    // CHECK: br %r{{.*}}
}
