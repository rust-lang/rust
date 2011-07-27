// -*- rust -*-
// xfail-stage0
// error-pattern: unresolved name: lt

fn f(a: int, b: int) { }

fn main() {
    let lt: int;
    let a: int = 10;
    let b: int = 23;
    check (lt(a, b));
    f(a, b);
}