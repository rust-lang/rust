//! regression test for <https://github.com/rust-lang/rust/issues/147415>
#![expect(incomplete_features)]
#![feature(min_generic_const_args)]

fn foo<T>() {
    [0; size_of::<*mut T>()];
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
    [0; const { size_of::<*mut T>() }];
    //~^ ERROR: generic parameters may not be used in const operations
    [0; const { size_of::<*mut i32>() }];
}

fn main() {}
