//@ check-pass

#[derive(Debug)]
struct S<T> {
    t: T,
    s: Box<S<fn(u: T)>>,
}

fn main() {}
