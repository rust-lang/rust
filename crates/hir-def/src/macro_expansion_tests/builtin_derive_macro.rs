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
        expect![[r##"
#[derive(Copy)]
struct Foo;

impl < > core::marker::Copy for Foo< > {}"##]],
    );
}

#[test]
fn test_copy_expand_in_core() {
    cov_mark::check!(test_copy_expand_in_core);
    check(
        r#"
#[rustc_builtin_macro]
macro derive {}
#[rustc_builtin_macro]
macro Copy {}
#[derive(Copy)]
struct Foo;
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro derive {}
#[rustc_builtin_macro]
macro Copy {}
#[derive(Copy)]
struct Foo;

impl < > crate ::marker::Copy for Foo< > {}"##]],
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
        expect![[r##"
#[derive(Copy)]
struct Foo<A, B>;

impl <T0: core::marker::Copy, T1: core::marker::Copy> core::marker::Copy for Foo<T0, T1> {}"##]],
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
        expect![[r##"
#[derive(Copy)]
struct Foo<A, B, 'a, 'b>;

impl <T0: core::marker::Copy, T1: core::marker::Copy> core::marker::Copy for Foo<T0, T1> {}"##]],
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
        expect![[r##"
#[derive(Clone)]
struct Foo<A, B>;

impl <T0: core::clone::Clone, T1: core::clone::Clone> core::clone::Clone for Foo<T0, T1> {}"##]],
    );
}
