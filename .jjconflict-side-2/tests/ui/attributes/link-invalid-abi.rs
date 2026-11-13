//@ check-fail
//@ compile-flags: --crate-type=lib

#![no_core]
#![feature(no_core)]
#![warn(unused_attributes)]

#[link(name = "first")]
//~^ WARN
//~| WARN
#[link(name = "second")]
extern "invalid" {}
//~^ ERROR
