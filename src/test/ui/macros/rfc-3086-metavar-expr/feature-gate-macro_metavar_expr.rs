// run-pass

#![feature(macro_metavar_expr)]

macro_rules! ignore {
    ( $( $i:ident ),* ) => {{
        let array: [i32; 0] = [$( ${ignore(i)} )*];
        array
    }};
}

fn main() {
    assert_eq!(ignore!(a, b, c), []);
}
