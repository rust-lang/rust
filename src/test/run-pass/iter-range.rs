

iter range(int a, int b) -> int {
    assert (a < b);
    let int i = a;
    while (i < b) { put i; i += 1; }
}

fn main() {
    let int sum = 0;
    for each (int x in range(0, 100)) { sum += x; }
    log sum;
}