

fn id[T](&T t) -> T { ret t; }

fn main() {
    auto t = rec(_0=1, _1=2, _2=3, _3=4, _4=5, _5=6, _6=7);
    assert (t._5 == 6);
    auto f1 = bind id[rec(int _0, int _1, int _2, int _3, int _4,
                          int _5, int _6)](_);
    assert (f1(t)._5 == 6);
}