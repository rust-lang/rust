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

fn main() { 0u32; }
"#]],
    );
}

#[test]
fn test_asm_expand() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! asm {() => {}}
#[rustc_builtin_macro]
macro_rules! global_asm {() => {}}
#[rustc_builtin_macro]
macro_rules! naked_asm {() => {}}

global_asm! {
    ""
}

#[unsafe(naked)]
extern "C" fn foo() {
    naked_asm!("");
}

fn main() {
    let i: u64 = 3;
    let o: u64;
    unsafe {
        asm!(
            "mov {0}, {1}",
            "add {0}, 5",
            out(reg) o,
            in(reg) i,
        );
    }
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! asm {() => {}}
#[rustc_builtin_macro]
macro_rules! global_asm {() => {}}
#[rustc_builtin_macro]
macro_rules! naked_asm {() => {}}

builtin #global_asm ("")

#[unsafe(naked)]
extern "C" fn foo() {
    builtin #naked_asm ("");
}

fn main() {
    let i: u64 = 3;
    let o: u64;
    unsafe {
        builtin #asm ("mov {0}, {1}", "add {0}, 5", out (reg)o, in (reg)i, );
    }
}
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
        expect![[r#"
#[rustc_builtin_macro]
macro_rules! line {() => {}}

fn main() { 0u32 }
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

fn main() { $crate::option::Option::None:: < &str>; }
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

fn main() { "file"; }
"##]],
    );
}

#[test]
fn test_assert_expand() {
    check(
        r#"
//- minicore: assert
fn main() {
    assert!(true, "{} {:?}", arg1(a, b, c), arg2);
}
"#,
        expect![[r#"
fn main() {
     {
        if !(true ) {
            $crate::panic::panic_2021!("{} {:?}", arg1(a, b, c), arg2);
        }
    };
}
"#]],
    );
}

// FIXME: This is the wrong expansion, see FIXME on `builtin_fn_macro::use_panic_2021`
#[test]
fn test_assert_expand_2015() {
    check(
        r#"
//- minicore: assert
//- /main.rs edition:2015
fn main() {
    assert!(true, "{} {:?}", arg1(a, b, c), arg2);
}
"#,
        expect![[r#"
fn main() {
     {
        if !(true ) {
            $crate::panic::panic_2021!("{} {:?}", arg1(a, b, c), arg2);
        }
    };
}
"#]],
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
    builtin #format_args ("{} {:?}", arg1(a, b, c), arg2);
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
    format_args!/*+errors*/(x =);
    format_args!/*+errors*/(x =, x = 2);
    format_args!/*+errors*/("{}", x =);
    format_args!/*+errors*/(=, "{}", x =);
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
    builtin #format_args (x = 2);
    /* parse error: expected expression */
builtin #format_args (x = );
    /* parse error: expected expression */
builtin #format_args (x = , x = 2);
    /* parse error: expected expression */
builtin #format_args ("{}", x = );
    /* parse error: expected expression */
/* parse error: expected expression */
builtin #format_args ( = , "{}", x = );
    builtin #format_args (x = 2, "{}", 5);
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
    builtin #format_args ("{} {:?}", a::<A, B>(), b);
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
    builtin #format_args (r#"{},mismatch,"{}","{}""#, location_csv_pat(db, &analysis, vfs, &sm, pat_id), mismatch.expected.display(db), mismatch.actual.display(db));
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
    builtin #format_args (concat!("xxx{}y", "{:?}zzz"), 2, b);
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
        /* parse error: expected field name or number */
builtin #format_args ("{} {:?}", a.);
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

fn main() { include_bytes("foo");include_bytes(r"foo"); }
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! include_bytes {
    ($file:expr) => {{ /* compiler built-in */ }};
    ($file:expr,) => {{ /* compiler built-in */ }};
}

fn main() { include_bytes("foo");include_bytes(r"foo"); }
"##]],
    );
}

#[test]
fn test_concat_expand() {
    check(
        r##"
#[rustc_builtin_macro]
macro_rules! concat {}

fn main() { concat!("fo", "o", 0, r#""bar""#, "\n", false, '"', -4, - 4, '\0'); }
"##,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! concat {}

fn main() { "foo0\"bar\"\nfalse\"-4-4\u{0}"; }
"##]],
    );
}

#[test]
fn test_concat_bytes_expand() {
    check(
        r##"
#[rustc_builtin_macro]
macro_rules! concat_bytes {}

fn main() { concat_bytes!(b'A', b"BC\"", [68, b'E', 70], br#"G""#,b'\0'); }
"##,
        expect![[r#"
#[rustc_builtin_macro]
macro_rules! concat_bytes {}

fn main() { b"ABC\"DEFG\"\x00"; }
"#]],
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
fn test_quote_string() {
    check(
        r##"
#[rustc_builtin_macro]
macro_rules! stringify {}

fn main() { stringify!("hello"); }
"##,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! stringify {}

fn main() { "\"hello\""; }
"##]],
    );
}

#[test]
fn cfg_select() {
    check(
        r#"
#[rustc_builtin_macro]
pub macro cfg_select($($tt:tt)*) {}

cfg_select! {
    false => { fn false_1() {} }
    any(false, true) => { fn true_1() {} }
}

cfg_select! {
    false => { fn false_2() {} }
    _ => { fn true_2() {} }
}

cfg_select! {
    false => { fn false_3() {} }
}

cfg_select! {
    false
}

cfg_select! {
    false =>
}

    "#,
        expect![[r#"
#[rustc_builtin_macro]
pub macro cfg_select($($tt:tt)*) {}

fn true_1() {}

fn true_2() {}

/* error: none of the predicates in this `cfg_select` evaluated to true */

/* error: expected `=>` after cfg expression */

/* error: expected a token tree after `=>` */

    "#]],
    );
}
