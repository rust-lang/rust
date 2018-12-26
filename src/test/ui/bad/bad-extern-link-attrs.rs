#[link()] //~ ERROR: specified without `name =
#[link(name = "")] //~ ERROR: with empty name
#[link(name = "foo")]
#[link(name = "foo", kind = "bar")] //~ ERROR: unknown kind
#[link] //~ ERROR #[link(...)] specified without arguments
#[link = "foo"] //~ ERROR #[link(...)] specified without arguments
extern {}

fn main() {}
