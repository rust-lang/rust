// Regression test for "attempted to remap an already remapped filename" ICE in rustdoc
// when using --remap-path-prefix with macro rendering.
// <https://github.com/rust-lang/rust/issues/138520>

//@ compile-flags:-Z unstable-options --remap-path-prefix={{src-base}}=remapped_path
//@ rustc-env:RUST_BACKTRACE=0
//@ build-pass

macro_rules! f(() => {});
