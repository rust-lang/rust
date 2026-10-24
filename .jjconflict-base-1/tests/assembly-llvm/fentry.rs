//@ assembly-output: emit-asm
//@ compile-flags: -Zinstrument-mcount=fentry
//@ add-minicore

//@ revisions: X86 S390X
//@[X86] compile-flags: --target=x86_64-unknown-linux-gnu -Cllvm-args=-x86-asm-syntax=intel
//@[X86] needs-llvm-components: x86
//@[S390X] compile-flags: --target=s390x-unknown-linux-gnu
//@[S390X] needs-llvm-components: systemz

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;

// CHECK-LABEL: mcount_func:
#[no_mangle]
pub fn mcount_func(a: isize, b: isize) -> isize {
    // X86: call __fentry__
    // S390X: brasl %r0, __fentry__@PLT
    a + b
    // X86: ret
    // S390X: br %r14
}
