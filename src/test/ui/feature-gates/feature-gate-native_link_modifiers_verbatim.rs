#[link(name = "foo", modifiers = "+verbatim")]
//~^ ERROR: linking modifier `verbatim` is unstable
extern "C" {}

fn main() {}
