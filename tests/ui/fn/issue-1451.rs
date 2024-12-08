//@ run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]
#![allow(unused_variables)]

struct T { f: extern "Rust" fn() }
struct S { f: extern "Rust" fn() }

fn fooS(t: S) {
}

fn fooT(t: T) {
}

fn bar() {
}

pub fn main() {
    let x: extern "Rust" fn() = bar;
    fooS(S {f: x});
    fooS(S {f: bar});

    let x: extern "Rust" fn() = bar;
    fooT(T {f: x});
    fooT(T {f: bar});
}
