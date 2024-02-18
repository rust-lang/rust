//@ run-pass
//@ pretty-expanded FIXME #23616

struct A(#[allow(dead_code)] bool);

pub fn main() {
    let f = A;
    f(true);
}
