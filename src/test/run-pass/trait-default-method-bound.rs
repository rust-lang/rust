trait A {
    fn g() -> int { 10 }
}

impl int: A { }

fn f<T: A>(i: T) {
    assert i.g() == 10;
}

fn main () {
    f(0);
}
