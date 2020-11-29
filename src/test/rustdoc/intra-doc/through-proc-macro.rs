// aux-build:through-proc-macro-aux.rs
// build-aux-docs
#![warn(broken_intra_doc_links)]
extern crate some_macros;

#[some_macros::second]
pub enum Boom {
    /// [Oooops]
    Bam,
}

fn main() {}
