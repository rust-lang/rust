// `#![no_std]` on a fully unconfigured crate is respected if it's placed before `cfg(FALSE)`.
// Therefore this crate does link to libstd.

#![cfg(FALSE)]
#![no_std]
