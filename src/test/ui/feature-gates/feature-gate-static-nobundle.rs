#[link(name="foo", kind="static-nobundle")]
//~^ ERROR: kind="static-nobundle" is feature gated
extern {}

fn main() {}
