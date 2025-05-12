//! Basic smoke test for `minicore` test auxiliary.
//!
//! This test is duplicated between ui/codegen/assembly because they have different runtest
//! codepaths.

//@ add-core-stubs
//@ check-pass
//@ compile-flags: --target=x86_64-unknown-linux-gnu
//@ needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

struct Meow;
impl Copy for Meow {}
