//@ run-pass

struct A(#[allow(dead_code)] bool);

pub fn main() {
    let f = A;
    f(true);
}
