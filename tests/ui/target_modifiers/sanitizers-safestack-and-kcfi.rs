//@ needs-sanitizer-kcfi
//@ needs-sanitizer-safestack

//@ aux-build:safestack-and-kcfi.rs
//@ compile-flags: -Cpanic=abort -Zunstable-options

//@ revisions: good good_reverted good_multiple missed_safestack missed_kcfi missed_both
//@[good] compile-flags: -Tsanitizer=safestack,kcfi
//@[good_reverted] compile-flags: -Tsanitizer=kcfi,safestack
//@[good_multiple] compile-flags: -Tsanitizer=safestack -Tsanitizer=kcfi
//@[missed_safestack] compile-flags: -Tsanitizer=kcfi
//@[missed_kcfi] compile-flags: -Tsanitizer=safestack
// [missed_both] no additional compile-flags:
//@[good] check-pass
//@[good_reverted] check-pass
//@[good_multiple] check-pass

#![feature(no_core)]
#![crate_type = "rlib"]
#![no_core]

extern crate safestack_and_kcfi;

//[missed_safestack]~? ERROR mixing `-Tsanitizer` will cause an ABI mismatch in crate `sanitizers_safestack_and_kcfi`
//[missed_kcfi]~? ERROR mixing `-Tsanitizer` will cause an ABI mismatch in crate `sanitizers_safestack_and_kcfi`
//[missed_both]~? ERROR mixing `-Tsanitizer` will cause an ABI mismatch in crate `sanitizers_safestack_and_kcfi`
