use std::any::Any;

struct Foo;

trait Bar {}

impl Bar for Foo {}

fn main() {
    let any: &dyn Any = &Bar; //~ ERROR expected value, found trait `Bar`
    if any.is::<u32>() { println!("u32"); }
}
