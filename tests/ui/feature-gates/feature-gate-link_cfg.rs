#[link(name = "foo", cfg(false))]
//~^ ERROR: is unstable
extern "C" {}

fn main() {}
