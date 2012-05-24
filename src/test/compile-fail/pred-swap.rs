// -*- rust -*-

// error-pattern:unsatisfied precondition constraint (for example, lt(a, b)

fn f(a: int, b: int) : lt(a, b) { }

pure fn lt(a: int, b: int) -> bool { ret a < b; }

fn main() {
    let mut a: int = 10;
    let mut b: int = 23;
    let mut c: int = 77;
    check (lt(a, b));
    b <-> a;
    f(a, b);
}
