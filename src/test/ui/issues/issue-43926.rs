#[link(name = "foo", cfg())] //~ ERROR link cfg must have a single predicate argument
extern "C" {}

fn main() {}
