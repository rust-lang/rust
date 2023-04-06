//! Tests for `builtin_derive_macro.rs` from `hir_expand`.

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn test_copy_expand_simple() {
    check(
        r#"
//- minicore: derive, copy
#[derive(Copy)]
struct Foo;
"#,
        expect![[r#"
#[derive(Copy)]
struct Foo;

impl < > core::marker::Copy for Foo< > where {}"#]],
    );
}

#[test]
fn test_copy_expand_in_core() {
    cov_mark::check!(test_copy_expand_in_core);
    check(
        r#"
//- /lib.rs crate:core
#[rustc_builtin_macro]
macro derive {}
#[rustc_builtin_macro]
macro Copy {}
#[derive(Copy)]
struct Foo;
"#,
        expect![[r#"
#[rustc_builtin_macro]
macro derive {}
#[rustc_builtin_macro]
macro Copy {}
#[derive(Copy)]
struct Foo;

impl < > crate ::marker::Copy for Foo< > where {}"#]],
    );
}

#[test]
fn test_copy_expand_with_type_params() {
    check(
        r#"
//- minicore: derive, copy
#[derive(Copy)]
struct Foo<A, B>;
"#,
        expect![[r#"
#[derive(Copy)]
struct Foo<A, B>;

impl <A: core::marker::Copy, B: core::marker::Copy, > core::marker::Copy for Foo<A, B, > where {}"#]],
    );
}

#[test]
fn test_copy_expand_with_lifetimes() {
    // We currently just ignore lifetimes
    check(
        r#"
//- minicore: derive, copy
#[derive(Copy)]
struct Foo<A, B, 'a, 'b>;
"#,
        expect![[r#"
#[derive(Copy)]
struct Foo<A, B, 'a, 'b>;

impl <A: core::marker::Copy, B: core::marker::Copy, > core::marker::Copy for Foo<A, B, > where {}"#]],
    );
}

#[test]
fn test_clone_expand() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
struct Foo<A, B>;
"#,
        expect![[r#"
#[derive(Clone)]
struct Foo<A, B>;

impl <A: core::clone::Clone, B: core::clone::Clone, > core::clone::Clone for Foo<A, B, > where {}"#]],
    );
}

#[test]
fn test_clone_expand_with_const_generics() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
struct Foo<const X: usize, T>(u32);
"#,
        expect![[r#"
#[derive(Clone)]
struct Foo<const X: usize, T>(u32);

impl <const X: usize, T: core::clone::Clone, > core::clone::Clone for Foo<X, T, > where u32: core::clone::Clone, {}"#]],
    );
}
