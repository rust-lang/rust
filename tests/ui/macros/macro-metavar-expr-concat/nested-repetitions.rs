//@ run-pass

#![feature(macro_metavar_expr_concat)]

macro_rules! two_layers {
    (
        $(
            [
                $outer:ident,
                (
                    $(
                        $inner:literal
                    ),*
                )
            ]
        ),*
    ) => {
        $(
            $(
                const ${concat($outer, $inner)}: i32 = 1;
            )*
        )*
    };
}

macro_rules! three_layers {
    (
        $(
            {
                $outer:tt,
                $(
                    [
                        $middle:ident,
                        (
                            $(
                                $inner:literal
                            ),*
                        )
                    ]
                ),*
            }
        ),*
    )
     => {
        $(
            $(
                $(
                    const ${concat($outer, $middle, $inner)}: i32 = 1;
                )*
            )*
        )*
    };
}

fn main() {
    two_layers!(
        [A_, ("FOO")],
        [B_, ("BAR", "BAZ")]
    );
    assert_eq!(A_FOO, 1);
    assert_eq!(B_BAR, 1);
    assert_eq!(B_BAZ, 1);

    three_layers!(
        {
            A_,
            [B_, ("FOO")],
            [C_, ("BAR", "BAZ")]
        },
        {
            D_,
            [E_, ("FOO")],
            [F_, ("BAR", "BAZ")]
        }
    );
    assert_eq!(A_B_FOO, 1);
    assert_eq!(A_C_BAR, 1);
    assert_eq!(A_C_BAZ, 1);
    assert_eq!(D_E_FOO, 1);
    assert_eq!(D_F_BAR, 1);
    assert_eq!(D_F_BAZ, 1);
}
