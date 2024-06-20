//@ run-pass

#![allow(dead_code, non_camel_case_types, non_upper_case_globals)]
#![feature(macro_metavar_expr_concat)]

macro_rules! create_things {
    ($lhs:ident) => {
        struct ${concat($lhs, _separated_idents_in_a_struct)} {
            foo: i32,
            ${concat($lhs, _separated_idents_in_a_field)}: i32,
        }

        mod ${concat($lhs, _separated_idents_in_a_module)} {
            pub const FOO: () = ();
        }

        fn ${concat($lhs, _separated_idents_in_a_fn)}() {}
    };
}

macro_rules! many_idents {
    ($a:ident, $c:ident) => {
        const ${concat($a, B, $c, D)}: i32 = 1;
    };
}

macro_rules! valid_tts {
    ($_0:tt, $_1:tt) => {
        const ${concat($_0, $_1)}: i32 = 1;
    }
}

macro_rules! without_dollar_sign_is_an_ident {
    ($ident:ident) => {
        const ${concat(VAR, ident)}: i32 = 1;
        const ${concat(VAR, $ident)}: i32 = 2;
    };
}

fn main() {
    create_things!(behold);
    behold_separated_idents_in_a_fn();
    let _ = behold_separated_idents_in_a_module::FOO;
    let _ = behold_separated_idents_in_a_struct {
        foo: 1,
        behold_separated_idents_in_a_field: 2,
    };

    many_idents!(A, C);
    assert_eq!(ABCD, 1);

    valid_tts!(X, YZ);
    assert_eq!(XYZ, 1);

    without_dollar_sign_is_an_ident!(_123);
    assert_eq!(VARident, 1);
    assert_eq!(VAR_123, 2);
}
