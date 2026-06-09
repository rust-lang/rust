// In rustc_hir_analysis::check::expr::no_such_field_err we recursively
// look in subfields for the field. This recursive search is limited
// in depth for compile-time reasons and to avoid infinite recursion
// in case of cycles. This file tests that the limit in the recursion
// depth is enforced.

struct Foo {
    first: Bar,
    second: u32,
    third: u32,
}

struct Bar {
    bar: C,
}

struct C {
    c: D,
}

struct D {
    test: E,
}

struct E {
    e: F,
}

struct F {
    f: u32,
}

fn main() {
    let f = F { f: 6 };
    let e = E { e: f };
    let d = D { test: e };
    let c = C { c: d };
    let bar = Bar { bar: c };
    let fooer = Foo { first: bar, second: 4, third: 5 };

    let test = fooer.f;
    //~^ ERROR no field
}
