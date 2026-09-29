//@ compile-flags: -Znext-solver
//@ build-fail
//@ failure-status: 101
//@ known-bug: #130516
//@ rustc-env:RUST_BACKTRACE=0
//@ normalize-stderr: "note: .*\n\n" -> ""
//@ normalize-stderr: "(compiler/[a-z_/]+\.rs):\d+:\d+" -> "$1:LL:CC"
//@ normalize-stderr: "query stack during panic:\n(.*\n)*?(end of query stack|\.\.\. and \d+ other queries.*)\n" -> ""

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
