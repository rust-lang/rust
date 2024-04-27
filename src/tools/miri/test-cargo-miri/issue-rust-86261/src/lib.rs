#![allow(unused_imports, unused_attributes, no_mangle_generic_items)]

// Regression test for https://github.com/rust-lang/rust/issues/86261:
// `#[no_mangle]` on a `use` item.
#[no_mangle]
use std::{any, boxed, io, panic, string, thread};

// `#[no_mangle]` on a struct has a similar problem.
#[no_mangle]
pub struct NoMangleStruct;

// If `#[no_mangle]` has effect on the `struct` above, calling `NoMangleStruct` will fail with
// "multiple definitions of symbol `NoMangleStruct`" error.
#[export_name = "NoMangleStruct"]
fn no_mangle_struct() {}

// `#[no_mangle]` on a generic function can also cause ICEs.
#[no_mangle]
fn no_mangle_generic<T>() {}

// Same as `no_mangle_struct()` but for the `no_mangle_generic()` generic function.
#[export_name = "no_mangle_generic"]
fn no_mangle_generic2() {}
