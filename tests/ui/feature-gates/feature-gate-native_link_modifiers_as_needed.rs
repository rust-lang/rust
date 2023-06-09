#[link(name = "foo", kind = "dylib", modifiers = "+as-needed")]
//~^ ERROR: linking modifier `as-needed` is unstable
extern "C" {}

fn main() {}
