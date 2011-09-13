// -*- rust -*-
// error-pattern: pure function calls function not known to be pure

fn g() { }

pure fn f(q: int) -> bool { g(); ret true; }

fn main() {
    let x = 0;

    check (f(x));
}
