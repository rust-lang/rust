//@ edition: 2021
//@ aux-build: extern-with-ambiguous-2-extern.rs

extern crate extern_with_ambiguous_2_extern;

mod s {
    pub mod error {
        use extern_with_ambiguous_2_extern::*;
    }
}
use s::*;
use extern_with_ambiguous_2_extern::*;
use error::*; //~ ERROR `error` is ambiguous

fn main() {}
