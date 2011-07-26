

fn box[T](&rec(T x, T y, T z) x) -> @rec(T x, T y, T z) { ret @x; }

fn main() {
    let @rec(int x, int y, int z) x = box[int](rec(x=1, y=2, z=3));
    assert (x.y == 2);
}