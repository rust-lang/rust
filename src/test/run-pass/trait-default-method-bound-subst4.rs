trait A<T> {
    fn g(x: uint) -> uint { move x }
}

impl<T> int: A<T> { }

fn f<T, V: A<T>>(i: V, j: uint) -> uint {
    i.g(move j)
}

fn main () {
    assert f::<float, int>(0, 2u) == 2u;
    assert f::<uint, int>(0, 2u) == 2u;
}
