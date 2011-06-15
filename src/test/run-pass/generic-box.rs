

fn box[T](&tup(T, T, T) x) -> @tup(T, T, T) { ret @x; }

fn main() {
    let @tup(int, int, int) x = box[int](tup(1, 2, 3));
    assert (x._1 == 2);
}