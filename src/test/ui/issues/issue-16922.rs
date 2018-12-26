use std::any::Any;

fn foo<T: Any>(value: &T) -> Box<Any> {
    Box::new(value) as Box<Any>
    //~^ ERROR explicit lifetime required in the type of `value` [E0621]
}

fn main() {
    let _ = foo(&5);
}
