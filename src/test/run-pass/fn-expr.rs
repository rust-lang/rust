


// xfail-stage0
fn main() { auto x = fn (int a) -> int { ret a + 1; }; assert (x(4) == 5); }