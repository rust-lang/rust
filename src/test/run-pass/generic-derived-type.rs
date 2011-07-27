

fn g[X](x: &X) -> X { ret x; }

fn f[T](t: &T) -> {a: T, b: T} {
    type pair = {a: T, b: T};

    let x: pair = {a: t, b: t};
    ret g[pair](x);
}

fn main() {
    let b = f[int](10);
    log b.a;
    log b.b;
    assert (b.a == 10);
    assert (b.b == 10);
}