#[link(name="foo", cfg("rlib"))] //~ ERROR invalid argument for `cfg(..)`
extern {}

fn main() {}
