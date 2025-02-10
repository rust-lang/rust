//@ aux-crate:wrong_regparm_and_ret=wrong_regparm_and_ret.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Cpanic=abort
// Auxiliary build problems with aarch64-apple:
// Shared library linking cc seems to convert "-m32" flag into -arch armv4t
// Auxiliary build problems with i686-mingw: linker `cc` not found
//@ only-x86
//@ ignore-windows
//@ ignore-apple
//@ needs-llvm-components: x86
//@ revisions:two_allowed unknown_allowed

//@[two_allowed] compile-flags: -Cunsafe-allow-abi-mismatch=regparm,reg-struct-return
//@[two_allowed] build-pass
//@[unknown_allowed] compile-flags: -Cunsafe-allow-abi-mismatch=unknown_flag -Zregparm=2 -Zreg-struct-return=true

#![crate_type = "rlib"]
//[unknown_allowed]~^ ERROR unknown target modifier `unknown_flag`, requested by `-Cunsafe-allow-abi-mismatch=unknown_flag`
#![no_core]
#![feature(no_core, lang_items, repr_simd)]

fn foo() {
    wrong_regparm_and_ret::somefun();
}
