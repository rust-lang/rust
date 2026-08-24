//! Regression test for <https://github.com/rust-lang/rust/issues/66367>.
//!
//! Adding `-C save-temps` to a follow-up incremental compile used to ICE: the codegen unit was
//! copied from the incremental cache, and the copy-from-cache path asserted that no bytecode was
//! wanted. `-C save-temps` wants it, so the assertion fired.

//@ revisions: bpass1 bpass2
//@ compile-flags: -Z query-dep-graph --crate-type=lib
//@[bpass2] compile-flags: -C save-temps

#![feature(rustc_attrs)]
// `-C save-temps` is not part of the incremental command line hash, so the codegen unit is still
// reused in `bpass2` -- which is the path that used to ICE.
#![rustc_partition_reused(module = "save_temps_66367", cfg = "bpass2")]

pub fn f() -> u32 {
    1
}
