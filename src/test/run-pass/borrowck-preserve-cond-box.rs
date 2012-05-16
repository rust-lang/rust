// xfail-fast  (compile-flags unsupported on windows)
// compile-flags:--borrowck=err
// exec-env:RUST_POISON_ON_FREE=1

fn testfn(cond: bool) {
    let mut x = @3;
    let mut y = @4;

    // borrow x and y
    let mut r_x = &*x;
    let mut r_y = &*y;
    let mut r = r_x, exp = 3;

    if cond {
        r = r_y;
        exp = 4;
    }

    #debug["*r = %d, exp = %d", *r, exp];
    assert *r == exp;

    x = @5;
    y = @6;

    #debug["*r = %d, exp = %d", *r, exp];
    assert *r == exp;
}

fn main() {
    testfn(true);
    testfn(false);
}