//@ run-pass
// Verifies that having generic parameters after constants is permitted
#[allow(dead_code)]
struct A<const N: usize, T>(T);

fn main() {}
