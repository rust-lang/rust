// xfail-test
type T = { mut f: fn@() };
type S = { f: fn@() };

fn fooS(t: S) {
}

fn fooT(t: T) {
}

fn bar() {
}

fn main() {
    let x: fn@() = bar;
    fooS({f: x});
    fooS({f: bar});

    let x: fn@() = bar;
    fooT({mut f: x});
    fooT({mut f: bar});
}

