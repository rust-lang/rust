// Illustrates the behavior of the `rustc_show_spans` internal attribute
// on field accesses.

//@ check-pass
#![feature(rustc_attrs)]

macro_rules! field {
    ($a:tt, $f:tt) => {
        let _ = $a.$f;
    };
}

const T: (u8,) = (1,);

#[rustc_show_spans(expr)]
fn field_access() {
    field!(T, 0);
}

struct S {
    x: u8,
}

const U: S = S { x: 1 };

#[rustc_show_spans(expr)]
fn named_field_access() {
    field!(U, x);
}

const V: ((u8,),) = ((1,),);

#[rustc_show_spans(expr)]
fn nested_field_access() {
    field!(V, 0.0);
}

fn main() {}
