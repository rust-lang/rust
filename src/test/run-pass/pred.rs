


// xfail-stage0
// -*- rust -*-
fn f(int a, int b) { }

pred lt(int a, int b) -> bool { ret a < b; }

fn main() {
    let int a = 10;
    let int b = 23;
    let int c = 77;
    check (lt(a, b));
    check (lt(a, c));
    f(a, b);
    f(a, c);
}