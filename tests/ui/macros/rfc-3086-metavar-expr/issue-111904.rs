#![feature(macro_metavar_expr)]

macro_rules! foo {
    ( $( $($t:ident),* );* ) => { ${count(t,)} }
    //~^ ERROR `count` followed by a comma must have an associated
    //~| ERROR expected expression, found `$`
}

fn test() {
    foo!(a, a; b, b);
}

fn main() {
}
