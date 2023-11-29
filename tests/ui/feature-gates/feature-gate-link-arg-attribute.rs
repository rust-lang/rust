#[link(kind = "link-arg", name = "foo")]
//~^ ERROR link kind `link-arg` is unstable
extern "C" {}

fn main() {}
