fn f(i: int, &called: bool) {
    assert i == 10;
    called = true;
}

fn g(f: fn(int, &bool), &called: bool) {
    f(10, called);
}

fn main() {
    let called = false;
    let h = f;
    g(h, called);
    assert called == true;
}