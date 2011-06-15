

fn g[X](&X x) -> X { ret x; }

fn f[T](&T t) -> tup(T, T) {
    type pair = tup(T, T);

    let pair x = tup(t, t);
    ret g[pair](x);
}

fn main() {
    auto b = f[int](10);
    log b._0;
    log b._1;
    assert (b._0 == 10);
    assert (b._1 == 10);
}