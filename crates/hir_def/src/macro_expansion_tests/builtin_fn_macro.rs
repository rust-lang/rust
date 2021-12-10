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
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! column {() => {}}

fn main() { 0; }
"##]],
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
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! line {() => {}}

fn main() { 0 }
"##]],
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

fn main() { "__RA_UNIMPLEMENTED__"; }
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
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! option_env {() => {}}

fn main() { std::option::Option::None:: < &str>; }
"##]],
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
        if !true {
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
compile_error!("error!");
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! compile_error {
    ($msg:expr) => ({ /* compiler built-in */ });
    ($msg:expr,) => ({ /* compiler built-in */ })
}

/* error: error! */
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
    unsafe {
        std::fmt::Arguments::new_v1(&[], &[std::fmt::ArgumentV1::new(&(arg1(a, b, c)), std::fmt::Display::fmt), std::fmt::ArgumentV1::new(&(arg2), std::fmt::Display::fmt), ])
    };
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
    unsafe {
        std::fmt::Arguments::new_v1(&[], &[std::fmt::ArgumentV1::new(&(a::<A, B>()), std::fmt::Display::fmt), std::fmt::ArgumentV1::new(&(b), std::fmt::Display::fmt), ])
    };
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
        // +errors
        format_args!("{} {:?}", a.);
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
        /* parse error: expected field name or number */
unsafe {
            std::fmt::Arguments::new_v1(&[], &[std::fmt::ArgumentV1::new(&(a.), std::fmt::Display::fmt), ])
        };
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

fn main() { concat!("foo", "r", 0, r#"bar"#, "\n", false); }
"##,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! concat {}

fn main() { "foor0bar\nfalse"; }
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

macro_rules! stuff {
    ($string:expr) => { concat!($string) };
}

fn main() { concat!(surprise!()); }
"##,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! concat {}

macro_rules! surprise {
    () => { "s" };
}

macro_rules! stuff {
    ($string:expr) => { concat!($string) };
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
