// run-pass
// pretty-expanded FIXME #23616

#[derive(Clone)]
struct S<T> {
    foo: (),
    bar: (),
    baz: T,
}

pub fn main() {
    let _ = S { foo: (), bar: (), baz: 1 }.clone();
}
