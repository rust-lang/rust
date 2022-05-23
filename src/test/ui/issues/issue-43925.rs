#[link(name = "foo", cfg("rlib"))] //~ ERROR link cfg must have a single predicate argument
extern "C" {}

fn main() {}
