// run-check
//@ proc-macro: proc-macro-type-error.rs
//@ ignore-backends: gcc

extern crate proc_macro_type_error;

use proc_macro_type_error::hello;

#[hello] //~ERROR mismatched types
fn abc() {}

fn x(_: &mut i32) {}

macro_rules! bla {
    () => {
        x(123);
        //~^ ERROR mismatched types
        //~| SUGGESTION &mut
    };
    ($v:expr) => {
        x($v)
    }
}

fn main() {
    bla!();
    bla!(456);
    //~^ ERROR mismatched types
    //~| SUGGESTION &mut
}
