// xfail-test

trait A<T> {
    fn g(x: T) -> T { move x }
}

impl int: A<int> { }

fn f<T, V: A<T>>(i: V, j: T) -> T {
    i.g(move j)
}

fn main () {
    assert f(0, 2) == 2;
}
