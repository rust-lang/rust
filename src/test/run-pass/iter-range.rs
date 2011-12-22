

fn range(a: int, b: int, it: block(int)) {
    assert (a < b);
    let i: int = a;
    while i < b { it(i); i += 1; }
}

fn main() {
    let sum: int = 0;
    range(0, 100) {|x| sum += x; }
    log_full(core::debug, sum);
}
