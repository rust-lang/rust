//@ add-minicore
//@ build-fail
//@ compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zcodegen-source-order
//@ needs-llvm-components: aarch64
//@ ignore-backends: gcc

// For pointer authentication the loader authenticates every entry of an init/fini array with the
// init/fini schema, but only functions whose ABI is supported (see `is_ptrauth_signed_fn`) can be
// signed.
// Reject non supported entries instead of emitting entries that fail authentication when the loader
// runs them.

#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

fn rust_init() {}

fn rust_fini() {}

extern "C" fn c_init() {}

#[used]
#[link_section = ".init_array"]
static RUST_INIT: fn() = rust_init;
//~^ ERROR `rust_init` is placed in `.init_array`, but pointer authentication does not sign its ABI

#[used]
#[link_section = ".fini_array.100"]
static RUST_FINI: [fn(); 3] = [rust_init, rust_fini, rust_fini];
//~^ ERROR `rust_init` is placed in `.fini_array.100`
//~| ERROR `rust_fini` is placed in `.fini_array.100`

#[used]
#[link_section = ".init_array"]
static C_INIT: extern "C" fn() = c_init;
