#[inline(never)]
pub fn main() {
    assert_eq!(fib(10), 55);
}

fn fib(n: usize) -> usize {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        let c = a;
        a = b;
        b = c + b;
    }
    a
}
