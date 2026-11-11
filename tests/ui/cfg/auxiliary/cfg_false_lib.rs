// `#![no_std]` on a fully unconfigured crate is respected if it's placed before `cfg(false)`.
// This crate has no such attribute, therefore this crate does link to libstd.

#![cfg(false)]
