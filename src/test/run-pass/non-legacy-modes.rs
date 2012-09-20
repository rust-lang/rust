struct X {
    repr: int
}

fn apply<T>(x: T, f: fn(T)) {
    f(x);
}

fn check_int(x: int) {
    assert x == 22;
}

fn check_struct(x: X) {
    check_int(x.repr);
}

fn main() {
    apply(22, check_int);
    apply(X {repr: 22}, check_struct);
}
