// Illustrates the behavior of the `rustc_show_spans` internal attribute
// on field accesses.

//@ check-pass
#![feature(rustc_attrs)]

// FIXME(#126763): the field access span should be `$a.$f` in the macro body.
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

fn main() {}
