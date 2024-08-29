//@ run-pass

#![feature(macro_metavar_expr_concat)]

macro_rules! one_rep {
    ( $($a:ident)* ) => {
        $(
            const ${concat($a, Z)}: i32 = 3;
        )*
    };
}

fn main() {
    one_rep!(A B C);
    assert_eq!(AZ, 3);
    assert_eq!(BZ, 3);
    assert_eq!(CZ, 3);
}
