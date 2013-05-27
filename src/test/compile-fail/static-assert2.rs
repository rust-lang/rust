#[static_assert]
static e: bool = 1 == 2; //~ ERROR static assertion failed

fn main() {}
