//@ check-pass
//@ compile-flags: --crate-type=lib

#![no_core]
#![feature(no_core, rust_cold_cc, rust_preserve_none_cc, rust_tail_cc, unboxed_closures)]
#![allow(missing_abi)]
#![deny(unfulfilled_lint_expectations, unused_attributes)]

#[link(name = "omitted")]
extern {}

#[link(name = "c")]
extern "C" {}

#[link(name = "rust-call")]
extern "rust-call" {}

#[link(name = "rust-cold")]
extern "rust-cold" {}

#[link(name = "rust-preserve-none")]
extern "rust-preserve-none" {}

#[link(name = "tail")]
extern "tail" {}

#[cfg_attr(any(), link(name = "disabled"))]
extern "Rust" {}

#[expect(unused_attributes)]
#[cfg_attr(all(), link(name = "enabled"))]
extern "Rust" {}

#[allow(unused_attributes)]
#[link(name = "allowed")]
extern "Rust" {}

#[expect(unused_attributes)]
#[link(name = "expected")]
extern "Rust" {}

#[expect(unused_attributes)]
#[link(name = "first")]
#[link(name = "second")]
extern "Rust" {}
