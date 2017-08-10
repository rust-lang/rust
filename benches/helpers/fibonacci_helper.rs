#[inline(never)]
pub fn main() {
    assert_eq!(fib(10), 55);
}

fn fib(n: usize) -> usize {
    if n <= 2 { 1 } else { fib(n - 1) + fib(n - 2) }
}
