//@ aux-crate:wrong_regparm=wrong_regparm.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Zregparm=1 -Cpanic=abort
//@ needs-llvm-components: x86
//@ revisions:error_generated allow_regparm_mismatch allow_no_value

//@[allow_regparm_mismatch] compile-flags: -Cunsafe-allow-abi-mismatch=regparm
//@[allow_regparm_mismatch] build-pass
//@[allow_no_value] compile-flags: -Cunsafe-allow-abi-mismatch

#![crate_type = "lib"]
//[error_generated]~^ ERROR mixing `-Zregparm` will cause an ABI mismatch in crate `incompatible_regparm`
#![no_core]
#![feature(no_core, lang_items, repr_simd)]

fn foo() {
    wrong_regparm::somefun();
}
