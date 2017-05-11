#![feature(plugin)]
#![plugin(clippy)]

#![deny(short_circuit_statement)]

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
