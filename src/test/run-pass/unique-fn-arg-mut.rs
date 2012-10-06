fn f(i: &mut ~int) {
    *i = ~200;
}

fn main() {
    let mut i = ~100;
    f(&mut i);
    assert *i == 200;
}