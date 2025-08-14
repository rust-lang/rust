// Test that Copy bounds inherited by trait are checked.


use std::any::Any;

trait Foo : Copy {
    fn foo(&self) {}
}

impl<T:Copy> Foo for T {
}

fn take_param<T:Foo>(foo: &T) { }

fn a() {
    let x: Box<_> = Box::new(3);
    take_param(&x); //~ ERROR E0277
}

fn b() {
    let x: Box<_> = Box::new(3);
    let y = &x;
    let z = &x as &dyn Foo;
    //~^ ERROR E0038
}

fn main() { }
