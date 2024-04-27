// `super` in a `macro` refers to the parent module of the macro itself and not its reexport.

//@ check-pass
//@ aux-build:macro-def-site-super.rs

extern crate macro_def_site_super;

type A = macro_def_site_super::public::mac!();

fn main() {}
