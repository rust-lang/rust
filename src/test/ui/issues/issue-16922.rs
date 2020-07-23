use std::any::Any;

fn foo<T: Any>(value: &T) -> Box<dyn Any> {
    Box::new(value) as Box<dyn Any>
    //~^ ERROR cannot infer an appropriate lifetime
}

fn main() {
    let _ = foo(&5);
}
