

fn range(a: int, b: int, it: fn(int)) {
    assert (a < b);
    let mut i: int = a;
    while i < b { it(i); i += 1; }
}

fn main() {
    let mut sum: int = 0;
    range(0, 100, {|x| sum += x; });
    log(debug, sum);
}
