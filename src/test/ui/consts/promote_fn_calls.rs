// build-pass (FIXME(62277): could be check-pass?)
// aux-build:promotable_const_fn_lib.rs

extern crate promotable_const_fn_lib;

use promotable_const_fn_lib::{foo, Foo};

fn main() {
    let x: &'static usize = &foo();
    let x: &'static usize = &Foo::foo();
}
