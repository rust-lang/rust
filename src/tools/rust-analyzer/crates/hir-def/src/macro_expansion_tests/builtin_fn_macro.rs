//! Tests for `builtin_fn_macro.rs` from `hir_expand`.

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn test_column_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! column {() => {}}

fn main() { column!(); }
"#,
        expect![[r#"
#[rustc_builtin_macro]
macro_rules! column {() => {}}

fn main() { 0 as u32; }
"#]],
    );
}

#[test]
fn test_line_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! line {() => {}}

fn main() { line!() }
"#,
        expect![[r#"
#[rustc_builtin_macro]
macro_rules! line {() => {}}

fn main() { 0 as u32 }
"#]],
    );
}

#[test]
fn test_stringify_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! stringify {() => {}}

fn main() {
    stringify!(
        a
        b
        c
    );
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! stringify {() => {}}

fn main() {
    "a b c";
}
"##]],
    );
}

#[test]
fn test_env_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! env {() => {}}

fn main() { env!("TEST_ENV_VAR"); }
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! env {() => {}}

fn main() { "UNRESOLVED_ENV_VAR"; }
"##]],
    );
}

#[test]
fn test_option_env_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! option_env {() => {}}

fn main() { option_env!("TEST_ENV_VAR"); }
"#,
        expect![[r#"
#[rustc_builtin_macro]
macro_rules! option_env {() => {}}

fn main() { ::core::option::Option::None:: < &str>; }
"#]],
    );
}

#[test]
fn test_file_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! file {() => {}}

fn main() { file!(); }
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! file {() => {}}

fn main() { ""; }
"##]],
    );
}

#[test]
fn test_assert_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! assert {
    ($cond:expr) => ({ /* compiler built-in */ });
    ($cond:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    assert!(true, "{} {:?}", arg1(a, b, c), arg2);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! assert {
    ($cond:expr) => ({ /* compiler built-in */ });
    ($cond:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
     {
        if !(true ) {
            $crate::panic!("{} {:?}", arg1(a, b, c), arg2);
        }
    };
}
"##]],
    );
}

#[test]
fn test_compile_error_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! compile_error {
    ($msg:expr) => ({ /* compiler built-in */ });
    ($msg:expr,) => ({ /* compiler built-in */ })
}

// This expands to nothing (since it's in item position), but emits an error.
compile_error!("error, with an escaped quote: \"");
compile_error!(r"this is a raw string");
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! compile_error {
    ($msg:expr) => ({ /* compiler built-in */ });
    ($msg:expr,) => ({ /* compiler built-in */ })
}

/* error: error, with an escaped quote: " */
/* error: this is a raw string */
"##]],
    );
}

#[test]
fn test_format_args_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    format_args!("{} {:?}", arg1(a, b, c), arg2);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    ::core::fmt::Arguments::new_v1(&["", " ", ], &[::core::fmt::Argument::new(&(arg1(a, b, c)), ::core::fmt::Display::fmt), ::core::fmt::Argument::new(&(arg2), ::core::fmt::Debug::fmt), ]);
}
"##]],
    );
}

#[test]
fn regression_15002() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    format_args!(x = 2);
    format_args!(x =);
    format_args!(x =, x = 2);
    format_args!("{}", x =);
    format_args!(=, "{}", x =);
    format_args!(x = 2, "{}", 5);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    /* error: no rule matches input tokens */;
    /* error: no rule matches input tokens */;
    /* error: no rule matches input tokens */;
    /* error: no rule matches input tokens */::core::fmt::Arguments::new_v1(&["", ], &[::core::fmt::Argument::new(&(), ::core::fmt::Display::fmt), ]);
    /* error: no rule matches input tokens */;
    ::core::fmt::Arguments::new_v1(&["", ], &[::core::fmt::Argument::new(&(5), ::core::fmt::Display::fmt), ]);
}
"##]],
    );
}

#[test]
fn test_format_args_expand_with_comma_exprs() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    format_args!("{} {:?}", a::<A,B>(), b);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    ::core::fmt::Arguments::new_v1(&["", " ", ], &[::core::fmt::Argument::new(&(a::<A, B>()), ::core::fmt::Display::fmt), ::core::fmt::Argument::new(&(b), ::core::fmt::Debug::fmt), ]);
}
"##]],
    );
}

#[test]
fn test_format_args_expand_with_raw_strings() {
    check(
        r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    format_args!(
        r#"{},mismatch,"{}","{}""#,
        location_csv_pat(db, &analysis, vfs, &sm, pat_id),
        mismatch.expected.display(db),
        mismatch.actual.display(db)
    );
}
"##,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    ::core::fmt::Arguments::new_v1(&[r#""#, r#",mismatch,""#, r#"",""#, r#"""#, ], &[::core::fmt::Argument::new(&(location_csv_pat(db, &analysis, vfs, &sm, pat_id)), ::core::fmt::Display::fmt), ::core::fmt::Argument::new(&(mismatch.expected.display(db)), ::core::fmt::Display::fmt), ::core::fmt::Argument::new(&(mismatch.actual.display(db)), ::core::fmt::Display::fmt), ]);
}
"##]],
    );
}

#[test]
fn test_format_args_expand_eager() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! concat {}

#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    format_args!(concat!("xxx{}y", "{:?}zzz"), 2, b);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! concat {}

#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    ::core::fmt::Arguments::new_v1(&["xxx", "y", "zzz", ], &[::core::fmt::Argument::new(&(2), ::core::fmt::Display::fmt), ::core::fmt::Argument::new(&(b), ::core::fmt::Debug::fmt), ]);
}
"##]],
    );
}

#[test]
fn test_format_args_expand_with_broken_member_access() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        format_args!/*+errors*/("{} {:?}", a.);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        /* error: no rule matches input tokens *//* parse error: expected field name or number */
::core::fmt::Arguments::new_v1(&["", " ", ], &[::core::fmt::Argument::new(&(a.), ::core::fmt::Display::fmt), ::core::fmt::Argument::new(&(), ::core::fmt::Debug::fmt), ]);
}
"##]],
    );
}

#[test]
fn test_include_bytes_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! include_bytes {
    ($file:expr) => {{ /* compiler built-in */ }};
    ($file:expr,) => {{ /* compiler built-in */ }};
}

fn main() { include_bytes("foo"); }
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! include_bytes {
    ($file:expr) => {{ /* compiler built-in */ }};
    ($file:expr,) => {{ /* compiler built-in */ }};
}

fn main() { include_bytes("foo"); }
"##]],
    );
}

#[test]
fn test_concat_expand() {
    check(
        r##"
#[rustc_builtin_macro]
macro_rules! concat {}

fn main() { concat!("foo", "r", 0, r#"bar"#, "\n", false, '"', '\0'); }
"##,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! concat {}

fn main() { "foor0bar\nfalse\"\u{0}"; }
"##]],
    );
}

#[test]
fn test_concat_bytes_expand() {
    check(
        r##"
#[rustc_builtin_macro]
macro_rules! concat_bytes {}

fn main() { concat_bytes!(b'A', b"BC", [68, b'E', 70]); }
"##,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! concat_bytes {}

fn main() { [b'A', 66, 67, 68, b'E', 70]; }
"##]],
    );
}

#[test]
fn test_concat_with_captured_expr() {
    check(
        r##"
#[rustc_builtin_macro]
macro_rules! concat {}

macro_rules! surprise {
    () => { "s" };
}

fn main() { concat!(surprise!()); }
"##,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! concat {}

macro_rules! surprise {
    () => { "s" };
}

fn main() { "s"; }
"##]],
    );
}

#[test]
fn test_concat_idents_expand() {
    check(
        r##"
#[rustc_builtin_macro]
macro_rules! concat_idents {}

fn main() { concat_idents!(foo, bar); }
"##,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! concat_idents {}

fn main() { foobar; }
"##]],
    );
}
