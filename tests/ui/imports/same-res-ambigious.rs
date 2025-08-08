//@ edition: 2018
//@ revisions: fail pass
//@[pass] aux-crate: ambigious_extern=same-res-ambigious-extern.rs
//@[fail] aux-crate: ambigious_extern=same-res-ambigious-extern-fail.rs
// see https://github.com/rust-lang/rust/pull/147196

#[derive(ambigious_extern::Embed)] //~ ERROR: derive macro `Embed` is private
struct Foo{}

fn main(){}
