//@ compile-flags: -Zforce-unstable-if-unmarked

#![feature(rustc_attrs)]

pub const fn not_stably_const() {}

#[rustc_const_stable_indirect]
pub const fn expose_on_stable() {}
