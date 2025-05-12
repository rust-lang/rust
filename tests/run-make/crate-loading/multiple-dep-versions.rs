extern crate dep_2_reexport;
extern crate dependency;
use dep_2_reexport::{OtherType, Trait2, Type};
use dependency::{Trait, do_something, do_something_trait, do_something_type};

fn main() {
    do_something(Type);
    Type.foo();
    Type::bar();
    do_something(OtherType);
    do_something_type(Type);
    do_something_trait(Box::new(Type) as Box<dyn Trait2>);
}
