//@ check-pass
//@ aux-build:through-proc-macro-aux.rs
//@ build-aux-docs

// Ensure rustdoc doesn't panic on this code.

#![warn(rustdoc::broken_intra_doc_links)]

extern crate some_macros;

#[some_macros::second]
pub enum Boom {
    /// [Oooops]
    //~^ WARNING unresolved link to `Oooops`
    Bam,
}

fn main() {}
