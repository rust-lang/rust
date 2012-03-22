fn f(&i: ~int) {
    i = ~200;
}

fn main() {
    let mut i = ~100;
    f(i);
    assert *i == 200;
}