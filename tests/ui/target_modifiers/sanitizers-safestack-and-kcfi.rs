//@ needs-sanitizer-kcfi
//@ needs-sanitizer-safestack

//@ aux-build:safestack-and-kcfi.rs
//@ compile-flags: -Cpanic=abort

//@ revisions: good good_reverted good_multiple missed_safestack missed_kcfi missed_both
//@[good] compile-flags: -Zsanitizer=safestack,kcfi
//@[good_reverted] compile-flags: -Zsanitizer=kcfi,safestack
//@[good_multiple] compile-flags: -Zsanitizer=safestack -Zsanitizer=kcfi
//@[missed_safestack] compile-flags: -Zsanitizer=kcfi
//@[missed_kcfi] compile-flags: -Zsanitizer=safestack
// [missed_both] no additional compile-flags:
//@[good] check-pass
//@[good_reverted] check-pass
//@[good_multiple] check-pass

#![feature(no_core)]
//[missed_safestack]~^ ERROR mixing `-Zsanitizer` will cause an ABI mismatch in crate `sanitizers_safestack_and_kcfi`
//[missed_kcfi]~^^ ERROR mixing `-Zsanitizer` will cause an ABI mismatch in crate `sanitizers_safestack_and_kcfi`
//[missed_both]~^^^ ERROR mixing `-Zsanitizer` will cause an ABI mismatch in crate `sanitizers_safestack_and_kcfi`
#![crate_type = "rlib"]
#![no_core]

extern crate safestack_and_kcfi;
