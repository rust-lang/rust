#[link(name = "foo", cfg("rlib"))]
//~^ ERROR link cfg is unstable
//~| ERROR `cfg` predicate key must be an identifier
extern "C" {}

fn main() {}
