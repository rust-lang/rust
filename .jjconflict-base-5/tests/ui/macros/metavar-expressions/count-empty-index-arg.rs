// Issue: https://github.com/rust-lang/rust/issues/111904
// Ensure that a trailing `,` is not interpreted as a `0`.

#![feature(macro_metavar_expr)]

macro_rules! foo {
    ( $( $($t:ident),* );* ) => { ${count($t,)} }
    //~^ ERROR `count` followed by a comma must have an associated
    //~| ERROR expected expression, found `$`
}

fn test() {
    foo!(a, a; b, b);
}

fn main() {}
