//! Basic smoke test for `minicore` test auxiliary.

//@ use-minicore
//@ revisions: meow
//@[meow] compile-flags: --target=x86_64-unknown-linux-gnu
//@[meow] needs-llvm-components: x86

#![crate_type = "lib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

struct Meow;
impl Copy for Meow {}

// CHECK-LABEL: meow
#[no_mangle]
fn meow() {}
