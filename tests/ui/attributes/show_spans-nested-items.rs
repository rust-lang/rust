// Illustrates the behavior of the `rustc_show_spans` internal attribute
// in the case of nested functions.

//@ check-pass
#![feature(rustc_attrs)]

#[rustc_show_spans(expr)]
fn outer() {
    // No note because it's a nested function.
    fn inner() {
        let _ = 0;
    }

    // The span will be over the `ty` (i.e, `u8`), not the `expr` (i.e., `1`).
    #[rustc_show_spans(ty)]
    fn inner_annotated() {
        let _: u8 = 1; //~ NOTE showing the spans of the ty
    }

    let _: u8 = 2; //~ NOTE showing the spans of the expr
}

// No note as the annotation of the previous function does not affect this one.
fn not_annotated() {
    let _ = 3;
}

fn main() {}
