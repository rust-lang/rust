// Test that we don't evaluate the initializer of free const items if the latter
// have non-region generic params (i.e., ones that "require monomorphization").
//@ revisions: fail pass
//@[pass] check-pass

#![feature(generic_const_items)]
#![expect(incomplete_features)]

const _<_T>: () = panic!();
const _<const _N: usize>: () = panic!();

#[cfg(fail)]
const _<'_a>: () = panic!(); //[fail]~ ERROR evaluation panicked: explicit panic

fn main() {}
