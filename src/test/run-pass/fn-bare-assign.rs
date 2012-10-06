fn f(i: int, called: &mut bool) {
    assert i == 10;
    *called = true;
}

fn g(f: extern fn(int, v: &mut bool), called: &mut bool) {
    f(10, called);
}

fn main() {
    let mut called = false;
    let h = f;
    g(h, &mut called);
    assert called == true;
}