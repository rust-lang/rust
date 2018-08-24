#[link(name = "foo", cfg(foo))]
//~^ ERROR: is feature gated
extern {}

fn main() {}
