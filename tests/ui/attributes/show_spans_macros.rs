// Illustrates the behavior of the `rustc_show_spans` internal attribute.
// It shows how the parser combines spans of nodes built from tokens of
// different macro contexts.
//
// The objective is for a node's span to be a chain over its internal
// members and not only from first to last. That is, we should converge
// to `span(first).to(span(second))...to(span(last))` instead of just
// `span(first).to(span(last))`. This way we can report spans inside
// macro definitions when the children come from different contexts.

//@ check-pass
#![feature(rustc_attrs)]

mod foo {
    pub const BAR: u8 = 1;
}

// FIXME(#126763): the span should be `$a::$b` in the macro body.
macro_rules! path {
    ($a:tt, $b:tt) => {
        let _ = $a::$b;
    };
}

#[rustc_show_spans(path)]
fn path_from_tts() {
    path!(foo, BAR);
}

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
