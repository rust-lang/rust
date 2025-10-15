#[link(name = "foo", cfg())] //~ ERROR malformed `link` attribute input
extern "C" {}

fn main() {}
