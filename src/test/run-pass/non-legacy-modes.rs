struct X {
    repr: isize
}

fn apply<T, F>(x: T, f: F) where F: FnOnce(T) {
    f(x);
}

fn check_int(x: isize) {
    assert_eq!(x, 22);
}

fn check_struct(x: X) {
    check_int(x.repr);
}

pub fn main() {
    apply(22, check_int);
    apply(X {repr: 22}, check_struct);
}
