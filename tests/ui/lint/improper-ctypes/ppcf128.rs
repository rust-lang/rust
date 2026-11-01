#![deny(improper_ctypes_definitions)]
//@ add-minicore
//@ check-pass
//@ revisions: PPC64 PPC64LE PPC32
//@ [PPC32] compile-flags: --target=powerpc-unknown-linux-gnu
//@ [PPC64] compile-flags: --target=powerpc64-unknown-linux-gnu
//@ [PPC64LE] compile-flags: --target=powerpc64le-unknown-linux-gnu
//@ needs-llvm-components: powerpc
#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

// The ppcf128 type is implemented as an ADT, so it needs custom logic
// to be accepted by improper_ctypes.

#[cfg(target_arch = "powerpc")]
use minicore::arch::powerpc::ppcf128;
#[cfg(target_arch = "powerpc64")]
use minicore::arch::powerpc64::ppcf128;
use minicore::*;

#[unsafe(no_mangle)]
extern "C" fn identity(x: ppcf128) -> ppcf128 {
    x
}
