#![feature(allow_internal_unstable)]
#![feature(bench_black_box)]
#![feature(extend_one)]
#![feature(iter_zip)]
#![feature(min_specialization)]
#![feature(test)]

pub mod bit_set;
pub mod vec;

// FIXME(#56935): Work around ICEs during cross-compilation.
#[allow(unused)]
extern crate rustc_macros;
