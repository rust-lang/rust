extern crate dep_2_reexport;
extern crate dependency;
use dep_2_reexport::{OtherType, Type};
use dependency::{Trait, do_something};

fn main() {
    do_something(Type);
    Type.foo();
    Type::bar();
    do_something(OtherType);
}
