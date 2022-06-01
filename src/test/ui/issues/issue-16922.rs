// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

use std::any::Any;

fn foo<T: Any>(value: &T) -> Box<dyn Any> {
    Box::new(value) as Box<dyn Any>
    //[base]~^ ERROR E0759
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() {
    let _ = foo(&5);
}
