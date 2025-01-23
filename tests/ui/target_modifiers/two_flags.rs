//@ aux-crate:wrong_regparm_and_ret=wrong_regparm_and_ret.rs
//@ compile-flags: --target i686-unknown-linux-gnu -Cpanic=abort
//@ needs-llvm-components: x86
//@ revisions:two_allowed unknown_allowed

//@[two_allowed] compile-flags: -Cunsafe-allow-abi-mismatch=regparm,reg-struct-return
//@[two_allowed] build-pass
//@[unknown_allowed] compile-flags: -Cunsafe-allow-abi-mismatch=unknown_flag -Zregparm=2 -Zreg-struct-return=true

#![crate_type = "lib"]
//[unknown_allowed]~^ ERROR unknown target modifier `unknown_flag`, requested by `-Cunsafe-allow-abi-mismatch=unknown_flag`
#![no_core]
#![feature(no_core, lang_items, repr_simd)]

fn foo() {
    wrong_regparm_and_ret::somefun();
}
