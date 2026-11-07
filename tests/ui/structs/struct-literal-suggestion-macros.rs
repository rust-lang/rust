//@ run-rustfix
// https://github.com/rust-lang/rust/issues/164044
#![allow(dead_code)]

struct NamedFields {
    b: Vec<u8>,
    c: usize,
}

fn main() {
    let _ = NamedFields(vec![], 2);
    //~^ ERROR cannot find function, tuple struct or tuple variant

    let b = vec![];
    let c = 2;
    let _ = NamedFields(b, c);
    //~^ ERROR cannot find function, tuple struct or tuple variant

    let b = vec![];
    let _ = NamedFields(b, 2);
    //~^ ERROR cannot find function, tuple struct or tuple variant

    let b = vec![];
    macro_rules! b_expr {
        () => { b };
    }

    let _ = NamedFields(b_expr!(), c);
    //~^ ERROR cannot find function, tuple struct or tuple variant
}
