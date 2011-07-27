// xfail-stage0
// -*- rust -*-

// error-pattern: Unsatisfied precondition constraint (for example, lt(a, b)

fn f(a: int, b: int) : lt(a,b) { }

pred lt(a: int, b: int) -> bool { ret a < b; }

fn main() {
    let a: int = 10;
    let b: int = 23;
    let c: int = 77;
    check (lt(a, b));
    a = 24;
    f(a, b);
}