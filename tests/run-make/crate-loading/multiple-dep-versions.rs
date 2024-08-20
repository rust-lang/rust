extern crate dep_2_reexport;
extern crate dependency;
use dep_2_reexport::Type;
use dependency::{do_something, Trait};

fn main() {
    do_something(Type);
    Type.foo();
    Type::bar();
}
