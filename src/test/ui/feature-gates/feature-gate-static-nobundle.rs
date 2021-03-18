#[link(name = "foo", kind = "static-nobundle")]
//~^ ERROR: kind="static-nobundle" is unstable
extern "C" {}

fn main() {}
