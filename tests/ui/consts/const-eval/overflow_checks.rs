//@ build-pass
//@ compile-flags: -O -C overflow-checks=no

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// Always returns true during CTFE, even if overflow checks are disabled.
const _: () = assert!(core::intrinsics::overflow_checks());
