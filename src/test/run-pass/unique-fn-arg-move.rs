fn f(-i: ~int) {
    assert *i == 100;
}

fn main() {
    let i = ~100;
    f(i);
}