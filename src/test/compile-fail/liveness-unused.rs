fn f1(x: int) {
    //!^ WARNING unused variable: `x`
}

fn f1b(&x: int) {
    //!^ WARNING unused variable: `x`
}

fn f2() {
    let x = 3;
    //!^ WARNING unused variable: `x`
}

fn f3() {
    let mut x = 3;
    //!^ WARNING variable `x` is assigned to, but never used
    x += 4;
    //!^ WARNING value assigned to `x` is never read
}

fn f3b() {
    let mut z = 3;
    //!^ WARNING variable `z` is assigned to, but never used
    loop {
        z += 4;
    }
}

fn f4() {
    alt some(3) {
      some(i) {
      }
      none {}
    }
}

// leave this in here just to trigger compile-fail:
pure fn is_even(i: int) -> bool { (i%2) == 0 }
fn even(i: int) : is_even(i) -> int { i }
fn main() {
    let i: int = 4;
    log(debug, false && { check is_even(i); true });
    even(i); //! ERROR unsatisfied precondition
}
