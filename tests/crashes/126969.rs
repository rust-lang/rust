//@ known-bug: rust-lang/rust#126969

struct S<T> {
    _: union { t: T },
}

fn f(S::<&i8> { .. }: S<&i8>) {}

fn main() {}
