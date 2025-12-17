//@ run-pass
#![feature(try_as_dyn)]

use std::any::try_as_dyn;

trait Trait {

}

impl Trait for for<'a> fn(&'a Box<i32>) {

}

fn store(_: &'static Box<i32>) {

}

fn main() {
    let fn_ptr: fn(&'static Box<i32>) = store;
    let dt = try_as_dyn::<_, dyn Trait>(&fn_ptr);
    assert!(dt.is_none());
}
