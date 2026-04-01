#[link(name = "foo", cfg("rlib"))]
//~^ ERROR link cfg is unstable
//~| ERROR malformed `link` attribute input
extern "C" {}

fn main() {}
