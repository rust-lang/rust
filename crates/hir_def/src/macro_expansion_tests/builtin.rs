//! Tests for builtin macros (see `builtin_macro.rs` in `hir_expand`).

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
