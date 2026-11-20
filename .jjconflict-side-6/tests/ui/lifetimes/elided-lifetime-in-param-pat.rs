//@ check-pass

struct S<T> {
    _t: T,
}

fn f(S::<&i8> { .. }: S<&i8>) {}

fn main() {
    f(S { _t: &42_i8 });
}
