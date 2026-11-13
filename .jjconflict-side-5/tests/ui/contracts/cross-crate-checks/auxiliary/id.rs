//@ [unchk_pass] compile-flags: -Zcontract-checks=no
//@ [unchk_fail] compile-flags: -Zcontract-checks=yes
//@ [chk_pass] compile-flags: -Zcontract-checks=no
//@ [chk_fail] compile-flags: -Zcontract-checks=yes

#![crate_type = "lib"]
#![feature(contracts)]

extern crate core;
use core::contracts::requires;

/// Example function with a spec to be called across a crate boundary.
#[requires(x > 0)]
pub fn id_if_positive(x: u32) -> u32 {
    x
}
