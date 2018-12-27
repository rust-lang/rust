// run-pass

fn f(i: isize, called: &mut bool) {
    assert_eq!(i, 10);
    *called = true;
}

fn g(f: fn(isize, v: &mut bool), called: &mut bool) {
    f(10, called);
}

pub fn main() {
    let mut called = false;
    let h = f;
    g(h, &mut called);
    assert_eq!(called, true);
}
