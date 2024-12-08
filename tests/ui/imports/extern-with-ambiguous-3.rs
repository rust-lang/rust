//@ check-pass
//@ edition: 2021
//@ aux-build: extern-with-ambiguous-3-extern.rs
// https://github.com/rust-lang/rust/pull/113099#issuecomment-1643974121

extern crate extern_with_ambiguous_3_extern;

mod s {
    pub mod error {
        use extern_with_ambiguous_3_extern::*;
    }
}
use s::*;
use extern_with_ambiguous_3_extern::*;
use error::*;
//^ FIXME: An ambiguity error should be thrown for `error`,
// as there is ambiguity present within `extern-with-ambiguous-3-extern.rs`.

fn main() {}
