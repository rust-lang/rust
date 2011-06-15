

fn id[T](&T t) -> T { ret t; }

fn main() {
    auto t = tup(1, 2, 3, 4, 5, 6, 7);
    assert (t._5 == 6);
    auto f0 = bind id[tup(int, int, int, int, int, int, int)](t);
    assert (f0()._5 == 6);
}