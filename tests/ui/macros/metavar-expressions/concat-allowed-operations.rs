//@ run-pass

#![allow(dead_code, non_camel_case_types, non_upper_case_globals, unused_variables)]
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

macro_rules! combinations {
    ($ident:ident, $literal:literal, $tt_ident:tt, $tt_literal:tt) => {{
        // tt ident
        let ${concat($tt_ident, b)} = ();
        let ${concat($tt_ident, _b)} = ();
        let ${concat($tt_ident, "b")} = ();
        let ${concat($tt_ident, $tt_ident)} = ();
        let ${concat($tt_ident, $tt_literal)} = ();
        let ${concat($tt_ident, $ident)} = ();
        let ${concat($tt_ident, $literal)} = ();
        // tt literal
        let ${concat($tt_literal, b)} = ();
        let ${concat($tt_literal, _b)} = ();
        let ${concat($tt_literal, "b")} = ();
        let ${concat($tt_literal, $tt_ident)} = ();
        let ${concat($tt_literal, $tt_literal)} = ();
        let ${concat($tt_literal, $ident)} = ();
        let ${concat($tt_literal, $literal)} = ();

        // ident (adhoc)
        let ${concat(_b, b)} = ();
        let ${concat(_b, _b)} = ();
        let ${concat(_b, "b")} = ();
        let ${concat(_b, $tt_ident)} = ();
        let ${concat(_b, $tt_literal)} = ();
        let ${concat(_b, $ident)} = ();
        let ${concat(_b, $literal)} = ();
        // ident (param)
        let ${concat($ident, b)} = ();
        let ${concat($ident, _b)} = ();
        let ${concat($ident, "b")} = ();
        let ${concat($ident, $tt_ident)} = ();
        let ${concat($ident, $tt_literal)} = ();
        let ${concat($ident, $ident)} = ();
        let ${concat($ident, $literal)} = ();

        // literal (adhoc)
        let ${concat("a", b)} = ();
        let ${concat("a", _b)} = ();
        let ${concat("a", "b")} = ();
        let ${concat("a", $tt_ident)} = ();
        let ${concat("a", $tt_literal)} = ();
        let ${concat("a", $ident)} = ();
        let ${concat("a", $literal)} = ();
        // literal (param)
        let ${concat($literal, b)} = ();
        let ${concat($literal, _b)} = ();
        let ${concat($literal, "b")} = ();
        let ${concat($literal, $tt_ident)} = ();
        let ${concat($literal, $tt_literal)} = ();
        let ${concat($literal, $ident)} = ();
        let ${concat($literal, $literal)} = ();
    }};
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

    combinations!(_hello, "a", b, "b");
}
