//@ check-pass
fn f(y: Box<isize>) {
    *y = 5; //~ WARNING cannot assign
}

fn g() {
    let _frob = |q: Box<isize>| { *q = 2; }; //~ WARNING cannot assign
}

fn main() {}
