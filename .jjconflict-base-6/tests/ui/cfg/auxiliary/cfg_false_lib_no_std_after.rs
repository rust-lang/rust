// `#![no_std]` on a fully unconfigured crate is respected if it's placed before `cfg(false)`.
// Therefore this crate does link to libstd.

#![cfg(false)]
#![no_std]
