//@ compile-flags: --crate-type rlib
//@ check-pass
//@ revisions: MB mb MiB MIB mib
//@[MB] rustc-env:RUST_MIN_STACK=16MB
//@[mb] rustc-env:RUST_MIN_STACK=16mb
//@[MiB] rustc-env:RUST_MIN_STACK=16MiB
//@[MIB] rustc-env:RUST_MIN_STACK=16MIB
//@[mib] rustc-env:RUST_MIN_STACK=16mib
