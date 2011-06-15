

type pair[T] = tup(T, T);

fn main() {
    let pair[int] x = tup(10, 12);
    assert (x._0 == 10);
    assert (x._1 == 12);
}