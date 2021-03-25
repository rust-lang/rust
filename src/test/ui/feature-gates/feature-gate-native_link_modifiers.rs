#[link(name = "foo", modifiers = "")]
//~^ ERROR: native link modifiers are experimental
extern "C" {}

fn main() {}
