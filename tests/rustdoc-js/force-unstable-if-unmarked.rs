//@ aux-crate:force_unstable_aux=force-unstable-aux.rs
//@ aux-build:force-unstable-aux.rs
//@ build-aux-docs
//@ compile-flags: -Zforce-unstable-if-unmarked
//@ edition:2021

#![feature(staged_api)]
#![crate_name = "force_unstable"]

pub fn std_path() {}
#[unstable(feature = "blob", reason = "blob", issue = "27721")]
pub fn std_path2() {}
