//! Check that `no_std` dynamic libraries can link and run without depending on `libstd`
//! now that there is an unconditional use of EII in `core`.

//@ build-pass
//@ compile-flags: -Cpanic=abort
//@ no-prefer-dynamic
//@ needs-crate-type: dylib
#![crate_type = "dylib"]
#![feature(core_io_internals)]
#![feature(core_io)]
#![feature(raw_os_error_ty)]
#![no_std]
#![no_implicit_prelude]

extern crate std;

pub use core::io::raw_os_error::*;
