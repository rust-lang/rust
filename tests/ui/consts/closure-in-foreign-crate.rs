// aux-build:closure-in-foreign-crate.rs
// build-pass

extern crate closure_in_foreign_crate;

const _: () = closure_in_foreign_crate::test();

fn main() {}
