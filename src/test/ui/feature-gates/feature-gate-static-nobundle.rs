#[link(name="foo", kind="static-nobundle")]
//~^ ERROR: #[link(kind="static-nobundle")] is feature gated
extern {}

fn main() {}
