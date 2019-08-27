#[link(name="foo", kind="raw-dylib")]
//~^ ERROR: kind="raw-dylib" is feature gated
extern {}

fn main() {}
