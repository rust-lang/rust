//@ run-pass

#[derive(Clone)]
#[allow(dead_code)]
struct S<T>(T, ());

pub fn main() {
    let _ = S(1, ()).clone();
}
