// test that we error correctly when effects is used without the next-solver flag.
//@ revisions: stock coherence full
//@[coherence] compile-flags: -Znext-solver=coherence
//@[full] compile-flags: -Znext-solver
//@[full] check-pass

#![feature(effects)]
#![allow(incomplete_features)]

fn main() {}
