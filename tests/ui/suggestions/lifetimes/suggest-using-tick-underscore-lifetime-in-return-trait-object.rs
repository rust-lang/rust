//@ run-rustfix
use std::any::Any;

fn foo<T: Any>(value: &T) -> Box<dyn Any> {
    Box::new(value) as Box<dyn Any>
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    let _ = foo(&5);
}
