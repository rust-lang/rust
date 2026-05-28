//@ run-pass

fn f<T:PartialEq + PartialOrd>(_: T) {
}

pub fn main() {
    f(3);
}
