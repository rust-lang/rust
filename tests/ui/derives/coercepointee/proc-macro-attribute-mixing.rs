// This test certify that we can mix attribute macros from Rust and external proc-macros.
// For instance, `#[derive(Default)]` uses `#[default]` and `#[derive(CoercePointee)]` uses
// `#[pointee]`.
// The scoping rule should allow the use of the said two attributes when external proc-macros
// are in scope.

//@ check-pass
//@ proc-macro: another-proc-macro.rs
//@ compile-flags: -Zunpretty=expanded
//@ edition: 2015

#![feature(derive_coerce_pointee)]

#[macro_use]
extern crate another_proc_macro;

#[pointee]
fn f() {}

#[default]
fn g() {}
