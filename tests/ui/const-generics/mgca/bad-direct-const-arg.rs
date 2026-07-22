//! Simple error message test, nothing special here
#![feature(min_generic_const_args)]

fn main(x: core::direct_const_arg!(2)) {
    //~^ ERROR expected type, found `direct_const_arg!()` constant
    let _ = core::direct_const_arg!(2);
    //~^ ERROR expected expression, found `direct_const_arg!()` constant
}
