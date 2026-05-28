//@ run-pass
// Tests that two closures can simultaneously have immutable
// access to the variable, whether that immutable access be used
// for direct reads or for taking immutable ref. Also check
// that the main function can read the variable too while
// the closures are in scope. Issue #6801.


fn a() -> i32 {
    let mut x = 3;
    x += 1;
    let c1 = || x * 4;
    let c2 = || x * 5;
    c1() * c2() * x
}

fn get(x: &i32) -> i32 {
    *x * 4
}

fn b() -> i32 {
    let mut x = 3;
    x += 1;
    let c1 = || get(&x);
    let c2 = || get(&x);
    c1() * c2() * x
}

fn c() -> i32 {
    let mut x = 3;
    x += 1;
    let c1 = || x * 5;
    let c2 = || get(&x);
    c1() * c2() * x
}

pub fn main() {
    assert_eq!(a(), 1280);
    assert_eq!(b(), 1024);
    assert_eq!(c(), 1280);
}
