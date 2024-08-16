extern crate dep_2_reexport;
extern crate dependency;
use dep_2_reexport::do_something;
use dependency::Type;

fn main() {
    do_something(Type);
}
