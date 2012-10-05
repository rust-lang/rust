fn f1(x: &mut int) {
    *x = 1; // no error
}

fn f2() {
    let mut x = 3; //~ WARNING value assigned to `x` is never read
    x = 4;
    copy x;
}

fn f3() {
    let mut x = 3;
    copy x;
    x = 4; //~ WARNING value assigned to `x` is never read
}

fn main() { // leave this in here just to trigger compile-fail:
    let x: int;
    copy x; //~ ERROR use of possibly uninitialized variable: `x`
}
