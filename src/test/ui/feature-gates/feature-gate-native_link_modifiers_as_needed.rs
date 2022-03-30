#[link(name = "foo", modifiers = "+as-needed")]
//~^ ERROR: `#[link(modifiers="as-needed")]` is unstable
extern "C" {}

fn main() {}
