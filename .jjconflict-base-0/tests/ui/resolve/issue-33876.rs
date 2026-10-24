use std::any::Any;

struct Foo;

trait Bar {}

impl Bar for Foo {}

fn main() {
    let any: &dyn Any = &Bar; //~ ERROR cannot find value `Bar` in this scope
    if any.is::<u32>() { println!("u32"); }
}
