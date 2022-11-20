#![warn(dead_code)]

struct Bar {
    #[allow(dead_code)]
    a: usize,
    #[forbid(dead_code)]
    b: usize, //~ ERROR field `b` is never read
    #[deny(dead_code)]
    c: usize, //~ ERROR fields `c` and `e` are never read
    d: usize, //~ WARN fields `d`, `f`, and `g` are never read
    #[deny(dead_code)]
    e: usize,
    f: usize,
    g: usize,
    _h: usize,
}

fn main() {
    Bar {
        a: 1,
        b: 1,
        c: 1,
        d: 1,
        e: 1,
        f: 1,
        g: 1,
        _h: 1,
    };
}
