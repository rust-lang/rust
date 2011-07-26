

fn g[X](&X x) -> X { ret x; }

fn f[T](&T t) -> rec(T a, T b) {
    type pair = rec(T a, T b);

    let pair x = rec(a=t, b=t);
    ret g[pair](x);
}

fn main() {
    auto b = f[int](10);
    log b.a;
    log b.b;
    assert (b.a == 10);
    assert (b.b == 10);
}