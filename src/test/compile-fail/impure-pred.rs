// -*- rust -*-
// error-pattern: Pure function calls function not known to be pure

fn g() { }

pred f(q: int) -> bool { g(); ret true; }

fn main() {
    let x = 0;

    check (f(x));
}
