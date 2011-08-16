

iter range(a: int, b: int) -> int {
    assert (a < b);
    let i: int = a;
    while i < b { put i; i += 1; }
}

fn main() {
    let sum: int = 0;
    for each x: int in range(0, 100) { sum += x; }
    log sum;
}
