fn f(i: ~int) {
    assert *i == 100;
}

fn main() {
    f(~100);
    let i = ~100;
    f(i);
}