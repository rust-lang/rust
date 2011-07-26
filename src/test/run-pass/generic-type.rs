

type pair[T] = rec(T x, T y);

fn main() {
    let pair[int] x = rec(x=10, y=12);
    assert (x.x == 10);
    assert (x.y == 12);
}