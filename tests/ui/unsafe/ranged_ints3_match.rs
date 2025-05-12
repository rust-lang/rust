#![feature(rustc_attrs)]

use std::cell::Cell;

#[rustc_layout_scalar_valid_range_start(1)]
#[repr(transparent)]
pub(crate) struct NonZero<T>(pub(crate) T);
fn main() {
    let mut x = unsafe { NonZero(Cell::new(1)) };
    match x {
        NonZero(ref x) => { x }
        //~^ ERROR borrow of layout constrained field with interior mutability
    };

    let mut y = unsafe { NonZero(42) };
    match y { NonZero(ref y) => { y } }; // OK, type of `y` is freeze
    match y { NonZero(ref mut y) => { y } };
    //~^ ERROR mutation of layout constrained field
}
