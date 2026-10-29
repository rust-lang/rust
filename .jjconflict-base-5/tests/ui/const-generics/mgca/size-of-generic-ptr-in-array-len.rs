//! regression test for <https://github.com/rust-lang/rust/issues/147415>
#![feature(gca_adts, gca_min_const_items, gca_macroless_args, min_adt_const_params)]

fn foo<T>() {
    [0; size_of::<*mut T>()];
    //~^ ERROR function items cannot be used as const args
    //~| ERROR tuple constructor with invalid base path
    [0; const { size_of::<*mut T>() }];
    //~^ ERROR: generic parameters may not be used in const operations
    [0; const { size_of::<*mut i32>() }];
}

fn main() {}
