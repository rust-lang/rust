

fn id[T](&T t) -> T { ret t; }

fn main() {
    auto t = rec(a=1, b=2, c=3, d=4, e=5, f=6, g=7);
    assert (t.f == 6);
    auto f0 = bind id(t);
    assert (f0().f == 6);
}