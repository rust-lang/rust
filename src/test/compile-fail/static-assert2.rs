#[allow(dead_code)];

#[static_assert]
static E: bool = 1 == 2; //~ ERROR static assertion failed

fn main() {}
