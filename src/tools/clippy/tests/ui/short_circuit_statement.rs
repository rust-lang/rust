#![warn(clippy::short_circuit_statement)]
#![allow(clippy::nonminimal_bool)]

fn main() {
    f() && g();
    f() || g();
    1 == 2 || g();
}

fn f() -> bool {
    true
}

fn g() -> bool {
    false
}
