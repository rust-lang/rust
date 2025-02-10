//@ aux-crate:wrong_regparm=wrong_regparm.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Zregparm=1 -Cpanic=abort
// Auxiliary build problems with aarch64-apple:
// Shared library linking cc seems to convert "-m32" flag into -arch armv4t
// Auxiliary build problems with i686-mingw: linker `cc` not found
//@ only-x86
//@ ignore-windows
//@ ignore-apple
//@ needs-llvm-components: x86
//@ revisions:error_generated allow_regparm_mismatch allow_no_value

//@[allow_regparm_mismatch] compile-flags: -Cunsafe-allow-abi-mismatch=regparm
//@[allow_regparm_mismatch] build-pass
//@[allow_no_value] compile-flags: -Cunsafe-allow-abi-mismatch

#![crate_type = "rlib"]
//[error_generated]~^ ERROR mixing `-Zregparm` will cause an ABI mismatch in crate `incompatible_regparm`
#![no_core]
#![feature(no_core, lang_items, repr_simd)]

fn foo() {
    wrong_regparm::somefun();
}
