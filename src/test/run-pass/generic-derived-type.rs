

fn g<X: copy>(x: X) -> X { return x; }

fn f<T: copy>(t: T) -> {a: T, b: T} {
    type pair = {a: T, b: T};

    let x: pair = {a: t, b: t};
    return g::<pair>(x);
}

fn main() {
    let b = f::<int>(10);
    log(debug, b.a);
    log(debug, b.b);
    assert (b.a == 10);
    assert (b.b == 10);
}
