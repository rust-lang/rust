#[link(name = "foo", cfg("rlib"))] //~ ERROR invalid argument for `cfg(..)`
extern "C" {}

fn main() {}
