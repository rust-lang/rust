// Top-level ill-formed
#[link] //~ ERROR malformed
#[link = "foo"] //~ ERROR malformed
extern "C" {}

fn main() {}
