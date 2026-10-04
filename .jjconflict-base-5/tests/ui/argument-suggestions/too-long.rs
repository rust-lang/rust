struct Qux;

impl Qux {
    fn foo(
        &self,
        a: i32,
        b: i32,
        c: i32,
        d: i32,
        e: i32,
        f: i32,
        g: i32,
        h: i32,
        i: i32,
        j: i32,
        k: i32,
        l: i32,
    ) {
    }
}

fn what(
    qux: &Qux,
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    e: i32,
    f: &i32,
    g: i32,
    h: i32,
    i: i32,
    j: i32,
    k: i32,
    l: i32,
) {
    qux.foo(a, b, c, d, e, f, g, h, i, j, k, l);
    //~^ ERROR mismatched types
}

fn main() {}
