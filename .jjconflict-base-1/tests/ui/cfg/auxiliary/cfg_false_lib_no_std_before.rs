// `#![no_std]` on a fully unconfigured crate is respected if it's placed before `cfg(false)`.
// Therefore this crate doesn't link to libstd.

//@ no-prefer-dynamic

#![no_std]
#![crate_type = "lib"]
#![cfg(false)]
