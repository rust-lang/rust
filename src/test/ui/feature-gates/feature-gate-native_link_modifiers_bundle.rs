#[link(name = "foo", kind = "static", modifiers = "+bundle")]
//~^ ERROR: linking modifier `bundle` is unstable
extern "C" {}

fn main() {}
