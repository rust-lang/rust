// -*- rust -*-
// error-pattern: Impure function as operator

fn f(q: int) -> bool { ret true; }

fn main() {
    let x = 0;

    check (f(x));
}
