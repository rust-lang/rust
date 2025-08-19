//@ run-pass

#![feature(macro_metavar_expr)]

macro_rules! nested {
    ( $a:ident ) => {
        macro_rules! $a {
            ( $$( $b:ident ),* ) => {
                $$(
                    macro_rules! $b {
                        ( $$$$( $c:ident ),* ) => {
                            $$$$(
                                fn $c() -> &'static str { stringify!($c) }
                            ),*
                        };
                    }
                )*
            };
        }
    };
}

fn main() {
    nested!(a);
    a!(b);
    b!(c);
    assert_eq!(c(), "c");
}
