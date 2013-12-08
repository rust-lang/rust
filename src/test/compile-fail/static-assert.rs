#[allow(dead_code)];

#[static_assert]
static A: bool = false; //~ ERROR static assertion failed

fn main() {
}
