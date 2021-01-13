#[link(name = "foo", cfg())] //~ ERROR `cfg()` must have an argument
extern "C" {}

fn main() {}
