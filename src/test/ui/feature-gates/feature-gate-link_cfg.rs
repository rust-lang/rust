#[link(name = "foo", cfg(foo))]
//~^ ERROR: is unstable
extern {}

fn main() {}
