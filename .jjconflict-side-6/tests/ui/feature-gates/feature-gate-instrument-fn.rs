//@ add-minicore
//@ compile-flags: --crate-type=rlib
#![no_core]
#![feature(no_core)]

extern crate minicore;
use minicore::*;

#[instrument_fn = "on"] //~ ERROR attribute is an experimental feature
fn instrumented_fn() {}

#[instrument_fn = "off"] //~ ERROR attribute is an experimental feature
fn not_instrumented_fn() {}
