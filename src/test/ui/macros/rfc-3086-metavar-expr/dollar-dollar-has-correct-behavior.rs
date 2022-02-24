// run-pass

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

macro_rules! not_nested {
    ( $$ $foo:ident ) => {{
        let $foo: i32 = 1;
        $foo
    }};
}


fn main() {
    nested!(a);
    a!(b);
    b!(c);
    assert_eq!(c(), "c");

    assert_eq!(not_nested!($foo), 1);
}
