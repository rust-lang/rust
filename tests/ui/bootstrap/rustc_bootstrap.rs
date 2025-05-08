//! Check the compiler's behavior when the perma-unstable env var `RUSTC_BOOTSTRAP` is set in the
//! environment in relation to feature stability and which channel rustc considers itself to be.
//!
//! `RUSTC_BOOTSTRAP` accepts:
//!
//! - `1`: cheat, allow usage of unstable features even if rustc thinks it is a stable compiler.
//! - `x,y,z`: comma-delimited list of crates.
//! - `-1`: force rustc to think it is a stable compiler.

// ignore-tidy-linelength

//@ revisions: default_nightly cheat cheat_single_crate cheat_multi_crate force_stable invalid_zero invalid_junk
//@ only-nightly

//@[default_nightly] unset-rustc-env:RUSTC_BOOTSTRAP
//@[default_nightly] check-pass

// For a nightly compiler, this is same as `default_nightly` as if `RUSTC_BOOTSTRAP` was unset.
//@[invalid_zero] rustc-env:RUSTC_BOOTSTRAP=0
//@[invalid_zero] check-pass

// Invalid values are silently discarded, same as `default_nightly`, i.e. as if `RUSTC_BOOTSTRAP`
// was unset.
//@[invalid_junk] rustc-env:RUSTC_BOOTSTRAP=*
//@[invalid_junk] check-pass

//@[cheat] rustc-env:RUSTC_BOOTSTRAP=1
//@[cheat] check-pass

//@[cheat_single_crate] rustc-env:RUSTC_BOOTSTRAP=x
//@[cheat_single_crate] check-pass

//@[cheat_multi_crate] rustc-env:RUSTC_BOOTSTRAP=x,y,z
//@[cheat_multi_crate] check-pass

// Note: compiletest passes some `-Z` flags to the compiler for ui testing purposes, so here we
// instead abuse the fact that `-Z unstable-options` is also part of rustc's stability story and is
// also affected by `RUSTC_BOOTSTRAP`.
//@[force_stable] rustc-env:RUSTC_BOOTSTRAP=-1
//@[force_stable] compile-flags: -Z unstable-options

#![crate_type = "lib"]

// Note: `rustc_attrs` is a perma-unstable internal feature that is unlikely to change, which is
// used as a proxy to check `RUSTC_BOOTSTRAP` versus stability checking logic.
#![feature(rustc_attrs)]

//[force_stable]~? RAW the option `Z` is only accepted on the nightly compiler
