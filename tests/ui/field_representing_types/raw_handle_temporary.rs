//@ revisions: old next
//@ [next] compile-flags: -Znext-solver
#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::raw_handle;

fn temp() -> i32 {
    42
}

fn main() {
    let _ = raw_handle!(42);
    //~^ ERROR: cannot take address of a temporary

    let _ = raw_handle!(temp());
    //~^ ERROR: cannot take address of a temporary
}
