// xfail-test

trait A<T> {
    fn g<U>(x: T, y: U) -> (T, U) { (move x, move y) }
}

impl int: A<int> { }

fn f<T, U, V: A<T>>(i: V, j: T, k: U) -> (T, U) {
    i.g(move j, move k)
}

fn main () {
    assert f(0, 1, 2) == (1, 2);
}
