#[link(name="foo", cfg())] //~ ERROR `cfg()` must have an argument
extern {}

fn main() {}
