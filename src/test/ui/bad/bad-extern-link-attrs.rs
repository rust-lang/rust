#[link()] //~ ERROR: specified without `name =
#[link(name = "")] //~ ERROR: with empty name
#[link(name = "foo")]
#[link(name = "foo", kind = "bar")] //~ ERROR: unknown kind
extern {}

fn main() {}
