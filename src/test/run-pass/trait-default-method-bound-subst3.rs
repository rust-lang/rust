trait A {
    fn g<T>(x: T, y: T) -> (T, T) { (move x, move y) }
}

impl int: A { }

fn f<T, V: A>(i: V, j: T, k: T) -> (T, T) {
    i.g(move j, move k)
}

fn main () {
    assert f(0, 1, 2) == (1, 2);
    assert f(0, 1u8, 2u8) == (1u8, 2u8);
}
