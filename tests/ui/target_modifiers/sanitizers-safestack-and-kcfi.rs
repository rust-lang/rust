//@ needs-sanitizer-kcfi
//@ needs-sanitizer-safestack

//@ aux-build:safestack-and-kcfi.rs
//@ compile-flags: -Cpanic=abort

//@ revisions: good good_reverted missed_safestack missed_kcfi missed_both
//@[good] compile-flags: -Zunstable-options -Csanitize=safestack,kcfi
//@[good_reverted] compile-flags: -Zunstable-options -Csanitize=kcfi,safestack
//@[missed_safestack] compile-flags: -Zunstable-options -Csanitize=kcfi
//@[missed_kcfi] compile-flags: -Zunstable-options -Csanitize=safestack
// [missed_both] no additional compile-flags:
//@[good] check-pass
//@[good_reverted] check-pass

#![feature(no_core)]
//[missed_safestack]~^ ERROR mixing `-Csanitize` will cause an ABI mismatch in crate `sanitizers_safestack_and_kcfi`
//[missed_kcfi]~^^ ERROR mixing `-Csanitize` will cause an ABI mismatch in crate `sanitizers_safestack_and_kcfi`
//[missed_both]~^^^ ERROR mixing `-Csanitize` will cause an ABI mismatch in crate `sanitizers_safestack_and_kcfi`
#![crate_type = "rlib"]
#![no_core]

extern crate safestack_and_kcfi;
