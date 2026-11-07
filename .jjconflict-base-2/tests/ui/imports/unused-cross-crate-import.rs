// issue: <https://github.com/rust-lang/rust/issues/12612>
// Test that unused `use` declarations involving multiple external crates are handled properly.
//@ run-pass
#![allow(unused_imports)]
//@ aux-build:unused-cross-crate-import-aux-1.rs
//@ aux-build:unused-cross-crate-import-aux-2.rs

extern crate unused_cross_crate_import_aux_1 as foo;
extern crate unused_cross_crate_import_aux_2 as bar;

mod test {
    use bar::baz;
}

fn main() {}
