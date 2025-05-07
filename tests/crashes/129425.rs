//@ known-bug: rust-lang/rust#129425

//@compile-flags: --crate-type=lib

#![feature(generic_const_exprs)]
fn foo<'a, T: 'a>(_: [(); std::mem::offset_of!((T,), 0)]) {}
