// This is a regression for `--remap-path-prefix` in an auxiliary dependency.
//
// We want to make sure that we can still have the "Source" links to the dependency
// even if its paths are remapped.
//
// See also rust-lang/rust#150100

//@ aux-build:remapped-paths.rs
//@ build-aux-docs

#![crate_name = "foo"]

extern crate remapped_paths;

//@ has foo/struct.MyStruct.html
//@ has - '//a[@href="../src/remapped_paths/remapped-paths.rs.html#3"]' 'Source'
//@ has - '//a[@href="../src/remapped_paths/remapped-paths.rs.html#8"]' 'Source'

pub use remapped_paths::MyStruct;
