// -*- rust -*-

// error-pattern: lt(a, c)

fn f(a: int, b: int) : lt(a, b) { }

pred lt(a: int, b: int) -> bool { ret a < b; }

fn main() {
    let a: int = 10;
    let b: int = 23;
    let c: int = 77;
    check (lt(a, b));
    check (lt(b, c));
    f(a, b);
    f(a, c);
}
