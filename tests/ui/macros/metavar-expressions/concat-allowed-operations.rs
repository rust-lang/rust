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
        #[allow(dead_code, reason = ${concat_str($a, B, $c, D)})]
        const ${concat($a, B, $c, D)}: i32 = 1;
    };
}

macro_rules! valid_tts {
    ($_0:tt, $_1:tt) => {
        #[allow(dead_code, reason = ${concat_str($_0, $_1)})]
        const ${concat($_0, $_1)}: i32 = 1;
    }
}

macro_rules! without_dollar_sign_is_an_ident {
    ($ident:ident) => {
        #[allow(dead_code, reason = ${concat_str(VAR, ident)})]
        const ${concat(VAR, ident)}: i32 = 1;
        #[allow(dead_code, reason = ${concat_str(VAR, $ident)})]
        const ${concat(VAR, $ident)}: i32 = 2;
    };
}

macro_rules! combinations {
    ($ident:ident, $literal:literal, $tt_ident:tt, $tt_literal:tt) => {{
        // tt ident
        #[allow(dead_code, reason = ${concat_str($tt_ident, b)})]
        let ${concat($tt_ident, b)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_ident, _b)})]
        let ${concat($tt_ident, _b)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_ident, "b")})]
        let ${concat($tt_ident, "b")} = ();
        #[allow(dead_code, reason = ${concat_str($tt_ident, $tt_ident)})]
        let ${concat($tt_ident, $tt_ident)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_ident, $tt_literal)})]
        let ${concat($tt_ident, $tt_literal)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_ident, $ident)})]
        let ${concat($tt_ident, $ident)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_ident, $ident)})]
        let ${concat($tt_ident, $ident)} = ();
        // tt literal
        #[allow(dead_code, reason = ${concat_str($tt_literal, b)})]
        let ${concat($tt_literal, b)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_literal, _b)})]
        let ${concat($tt_literal, _b)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_literal, "b")})]
        let ${concat($tt_literal, "b")} = ();
        #[allow(dead_code, reason = ${concat_str($tt_literal, $tt_ident)})]
        let ${concat($tt_literal, $tt_ident)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_literal, $tt_literal)})]
        let ${concat($tt_literal, $tt_literal)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_literal, $ident)})]
        let ${concat($tt_literal, $ident)} = ();
        #[allow(dead_code, reason = ${concat_str($tt_literal, $literal)})]
        let ${concat($tt_literal, $literal)} = ();

        // ident (adhoc)
        #[allow(dead_code, reason = ${concat_str(_b, b)})]
        let ${concat(_b, b)} = ();
        #[allow(dead_code, reason = ${concat_str(_b, _b)})]
        let ${concat(_b, _b)} = ();
        #[allow(dead_code, reason = ${concat_str(_b, "b")})]
        let ${concat(_b, "b")} = ();
        #[allow(dead_code, reason = ${concat_str(_b, $tt_ident)})]
        let ${concat(_b, $tt_ident)} = ();
        #[allow(dead_code, reason = ${concat_str(_b, $tt_literal)})]
        let ${concat(_b, $tt_literal)} = ();
        #[allow(dead_code, reason = ${concat_str(_b, $ident)})]
        let ${concat(_b, $ident)} = ();
        #[allow(dead_code, reason = ${concat_str(_b, $literal)})]
        let ${concat(_b, $literal)} = ();
        // ident (param)
        #[allow(dead_code, reason = ${concat_str($ident, b)})]
        let ${concat($ident, b)} = ();
        #[allow(dead_code, reason = ${concat_str($ident, _b)})]
        let ${concat($ident, _b)} = ();
        #[allow(dead_code, reason = ${concat_str($ident, "b")})]
        let ${concat($ident, "b")} = ();
        #[allow(dead_code, reason = ${concat_str($ident, $tt_ident)})]
        let ${concat($ident, $tt_ident)} = ();
        #[allow(dead_code, reason = ${concat_str($ident, $tt_literal)})]
        let ${concat($ident, $tt_literal)} = ();
        #[allow(dead_code, reason = ${concat_str($ident, $ident)})]
        let ${concat($ident, $ident)} = ();
        #[allow(dead_code, reason = ${concat_str($ident, $literal)})]
        let ${concat($ident, $literal)} = ();

        // literal (adhoc)
        #[allow(dead_code, reason = ${concat_str("a", b)})]
        let ${concat("a", b)} = ();
        #[allow(dead_code, reason = ${concat_str("a", _b)})]
        let ${concat("a", _b)} = ();
        #[allow(dead_code, reason = ${concat_str("a", "b")})]
        let ${concat("a", "b")} = ();
        #[allow(dead_code, reason = ${concat_str("a", $tt_ident)})]
        let ${concat("a", $tt_ident)} = ();
        #[allow(dead_code, reason = ${concat_str("a", $tt_literal)})]
        let ${concat("a", $tt_literal)} = ();
        #[allow(dead_code, reason = ${concat_str("a", $ident)})]
        let ${concat("a", $ident)} = ();
        #[allow(dead_code, reason = ${concat_str("a", $literal)})]
        let ${concat("a", $literal)} = ();
        // literal (param)
        #[allow(dead_code, reason = ${concat_str($literal, b)})]
        let ${concat($literal, b)} = ();
        #[allow(dead_code, reason = ${concat_str($literal, _b)})]
        let ${concat($literal, _b)} = ();
        #[allow(dead_code, reason = ${concat_str($literal, "b")})]
        let ${concat($literal, "b")} = ();
        #[allow(dead_code, reason = ${concat_str($literal, $tt_ident)})]
        let ${concat($literal, $tt_ident)} = ();
        #[allow(dead_code, reason = ${concat_str($literal, $tt_literal)})]
        let ${concat($literal, $tt_literal)} = ();
        #[allow(dead_code, reason = ${concat_str($literal, $ident)})]
        let ${concat($literal, $ident)} = ();
        #[allow(dead_code, reason = ${concat_str($literal, $literal)})]
        let ${concat($literal, $literal)} = ();
    }};
}

macro_rules! int_struct {
    ($n: literal) => {
        #[allow(dead_code, reason = ${concat_str(E, $n)})]
        struct ${concat(E, $n)};
    }
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

    int_struct!(1_0);
    int_struct!(2);
    int_struct!(3___0);
    int_struct!(7_);
    int_struct!(08);

    let _ = E1_0;
    let _ = E2;
    let _ = E3___0;
    let _ = E7_;
    let _ = E08;
}
