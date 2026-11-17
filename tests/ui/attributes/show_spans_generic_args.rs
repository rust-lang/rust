// Illustrates the behavior of the `rustc_show_spans` internal attribute
// on generic arguments.

//@ check-pass
#![feature(rustc_attrs)]

#[rustc_show_spans(generic_args)]
fn angle_bracketed() {
    let _: Vec<u8>;
    let _: Vec<Option<u8>>;
    let _ = Vec::<u8>::new();
}

macro_rules! parenthesized {
    ($f:ident) => {
        let _: &dyn $f(u8) -> u8;
    };
}

#[rustc_show_spans(generic_args)]
fn parenthesized_from_ident() {
    parenthesized!(Fn);
}

macro_rules! arrow {
    ($f:tt $args:tt, $o:tt) => {
        let _: &dyn $f $args -> $o;
    };
}

#[rustc_show_spans(path)]
fn arrow_from_macro() {
    arrow!(Fn (u8), u8);
}

fn main() {}
