// aux-build:through-proc-macro-aux.rs
// build-aux-docs
#![warn(intra_doc_link_resolution_failure)]
extern crate some_macros;

#[some_macros::second]
pub enum Boom {
    /// [Oooops]
    Bam,
}

fn main() {}
