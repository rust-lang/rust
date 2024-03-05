#![allow(incomplete_features)]
#![feature(unnamed_fields)]

#[repr(C)]
#[derive(Clone, Copy)]
struct Foo {
    x: (),
    y: (),
}

#[repr(C)]
#[derive(Clone, Copy)]
union Bar {
    w: (),
    z: (),
}

#[repr(C)]
#[derive(Clone, Copy)]
struct U {
    a: (),
    _: struct {
        b: (),
        c: (),
        _: union {
            d: (),
            e: (),
        },
        _: union {},
    },
    _: union {
        f: (),
        g: (),
        _: struct {
            i: (),
            j: (),
        },
        _: struct {},
    },
    _: Foo,
    _: Bar,
}

#[repr(C)]
#[derive(Clone, Copy)]
union V {
    a: (),
    _: struct {
        b: (),
        c: (),
        _: union {
            d: (),
            e: (),
        },
        _: union {},
    },
    _: union {
        f: (),
        g: (),
        _: struct {
            i: (),
            j: (),
        },
        _: struct {},
    },
    _: Foo,
    _: Bar,
}

fn case_u(u: U) {
    let U { a, b, c, d, e, f, g, i, j, x, y, w, z } = u; //~ ERROR union patterns should have exactly one field
}

fn case_v(v: V) {
    let V { a, b, c, d, e, f, g, i, j, x, y, w, z } = v; //~ ERROR union patterns should have exactly one field
}

fn main() {}
