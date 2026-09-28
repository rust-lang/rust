//@ compile-flags: -Znext-solver
//@ build-fail
//@ known-bug: #130516

// Impls for unsafe binders should have the correct def-path.

#![feature(unsafe_binders, rustc_attrs)]
#![allow(incomplete_features)]
#![crate_type = "lib"]

pub struct Local;

pub mod t {
    pub trait Tr {
        fn m(&self);
    }
}

impl t::Tr for unsafe<'a> &'a Local {
    #[rustc_dump_def_path]
    fn m(&self) {}
}

impl t::Tr for &Local {
    #[rustc_dump_def_path]
    fn m(&self) {}
}
