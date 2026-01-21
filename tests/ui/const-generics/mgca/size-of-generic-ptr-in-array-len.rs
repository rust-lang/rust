//! regression test for <https://github.com/rust-lang/rust/issues/147415>
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

fn foo<T>() {
    [0; size_of::<*mut T>()];
    //~^ ERROR: tuple constructor with invalid base path
    [0; const { size_of::<*mut T>() }];
    //~^ ERROR: generic parameters may not be used in const operations
}

fn main() {}
