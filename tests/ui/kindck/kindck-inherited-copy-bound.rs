// Test that Copy bounds inherited by trait are checked.
//
// revisions: curr object_safe_for_dispatch

#![cfg_attr(object_safe_for_dispatch, feature(object_safe_for_dispatch))]


use std::any::Any;

trait Foo : Copy {
    fn foo(&self) {}
}

impl<T:Copy> Foo for T {
}

fn take_param<T:Foo>(foo: &T) { }

fn a() {
    let x: Box<_> = Box::new(3);
    take_param(&x); //[curr]~ ERROR E0277
    //[object_safe_for_dispatch]~^ ERROR E0277
}

fn b() {
    let x: Box<_> = Box::new(3);
    let y = &x;
    let z = &x as &dyn Foo;
    //[curr]~^ ERROR E0038
    //[curr]~| ERROR E0038
    //[object_safe_for_dispatch]~^^^ ERROR E0038
}

fn main() { }
