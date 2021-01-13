#[link(name = "foo", cfg(foo))]
//~^ ERROR: is unstable
extern "C" {}

fn main() {}
