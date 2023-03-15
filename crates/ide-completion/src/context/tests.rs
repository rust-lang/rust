use expect_test::{expect, Expect};
use hir::HirDisplay;

use crate::{
    context::CompletionContext,
    tests::{position, TEST_CONFIG},
};

fn check_expected_type_and_name(ra_fixture: &str, expect: Expect) {
    let (db, pos) = position(ra_fixture);
    let config = TEST_CONFIG;
    let (completion_context, _analysis) = CompletionContext::new(&db, pos, &config).unwrap();

    let ty = completion_context
        .expected_type
        .map(|t| t.display_test(&db).to_string())
        .unwrap_or("?".to_owned());

    let name =
        completion_context.expected_name.map_or_else(|| "?".to_owned(), |name| name.to_string());

    expect.assert_eq(&format!("ty: {ty}, name: {name}"));
}

#[test]
fn expected_type_let_without_leading_char() {
    cov_mark::check!(expected_type_let_without_leading_char);
    check_expected_type_and_name(
        r#"
fn foo() {
    let x: u32 = $0;
}
"#,
        expect![[r#"ty: u32, name: x"#]],
    );
}

#[test]
fn expected_type_let_with_leading_char() {
    cov_mark::check!(expected_type_let_with_leading_char);
    check_expected_type_and_name(
        r#"
fn foo() {
    let x: u32 = c$0;
}
"#,
        expect![[r#"ty: u32, name: x"#]],
    );
}

#[test]
fn expected_type_let_pat() {
    check_expected_type_and_name(
        r#"
fn foo() {
    let x$0 = 0u32;
}
"#,
        expect![[r#"ty: u32, name: ?"#]],
    );
    check_expected_type_and_name(
        r#"
fn foo() {
    let $0 = 0u32;
}
"#,
        expect![[r#"ty: u32, name: ?"#]],
    );
}

#[test]
fn expected_type_fn_param() {
    cov_mark::check!(expected_type_fn_param);
    check_expected_type_and_name(
        r#"
fn foo() { bar($0); }
fn bar(x: u32) {}
"#,
        expect![[r#"ty: u32, name: x"#]],
    );
    check_expected_type_and_name(
        r#"
fn foo() { bar(c$0); }
fn bar(x: u32) {}
"#,
        expect![[r#"ty: u32, name: x"#]],
    );
}

#[test]
fn expected_type_fn_param_ref() {
    cov_mark::check!(expected_type_fn_param_ref);
    check_expected_type_and_name(
        r#"
fn foo() { bar(&$0); }
fn bar(x: &u32) {}
"#,
        expect![[r#"ty: u32, name: x"#]],
    );
    check_expected_type_and_name(
        r#"
fn foo() { bar(&mut $0); }
fn bar(x: &mut u32) {}
"#,
        expect![[r#"ty: u32, name: x"#]],
    );
    check_expected_type_and_name(
        r#"
fn foo() { bar(& c$0); }
fn bar(x: &u32) {}
        "#,
        expect![[r#"ty: u32, name: x"#]],
    );
    check_expected_type_and_name(
        r#"
fn foo() { bar(&mut c$0); }
fn bar(x: &mut u32) {}
"#,
        expect![[r#"ty: u32, name: x"#]],
    );
    check_expected_type_and_name(
        r#"
fn foo() { bar(&c$0); }
fn bar(x: &u32) {}
        "#,
        expect![[r#"ty: u32, name: x"#]],
    );
}

#[test]
fn expected_type_struct_field_without_leading_char() {
    cov_mark::check!(expected_type_struct_field_without_leading_char);
    check_expected_type_and_name(
        r#"
struct Foo { a: u32 }
fn foo() {
    Foo { a: $0 };
}
"#,
        expect![[r#"ty: u32, name: a"#]],
    )
}

#[test]
fn expected_type_struct_field_followed_by_comma() {
    cov_mark::check!(expected_type_struct_field_followed_by_comma);
    check_expected_type_and_name(
        r#"
struct Foo { a: u32 }
fn foo() {
    Foo { a: $0, };
}
"#,
        expect![[r#"ty: u32, name: a"#]],
    )
}

#[test]
fn expected_type_generic_struct_field() {
    check_expected_type_and_name(
        r#"
struct Foo<T> { a: T }
fn foo() -> Foo<u32> {
    Foo { a: $0 }
}
"#,
        expect![[r#"ty: u32, name: a"#]],
    )
}

#[test]
fn expected_type_struct_field_with_leading_char() {
    cov_mark::check!(expected_type_struct_field_with_leading_char);
    check_expected_type_and_name(
        r#"
struct Foo { a: u32 }
fn foo() {
    Foo { a: c$0 };
}
"#,
        expect![[r#"ty: u32, name: a"#]],
    );
}

#[test]
fn expected_type_match_arm_without_leading_char() {
    cov_mark::check!(expected_type_match_arm_without_leading_char);
    check_expected_type_and_name(
        r#"
enum E { X }
fn foo() {
   match E::X { $0 }
}
"#,
        expect![[r#"ty: E, name: ?"#]],
    );
}

#[test]
fn expected_type_match_arm_with_leading_char() {
    cov_mark::check!(expected_type_match_arm_with_leading_char);
    check_expected_type_and_name(
        r#"
enum E { X }
fn foo() {
   match E::X { c$0 }
}
"#,
        expect![[r#"ty: E, name: ?"#]],
    );
}

#[test]
fn expected_type_match_arm_body_without_leading_char() {
    cov_mark::check!(expected_type_match_arm_body_without_leading_char);
    check_expected_type_and_name(
        r#"
struct Foo;
enum E { X }
fn foo() -> Foo {
   match E::X { E::X => $0 }
}
"#,
        expect![[r#"ty: Foo, name: ?"#]],
    );
}

#[test]
fn expected_type_match_body_arm_with_leading_char() {
    cov_mark::check!(expected_type_match_arm_body_with_leading_char);
    check_expected_type_and_name(
        r#"
struct Foo;
enum E { X }
fn foo() -> Foo {
   match E::X { E::X => c$0 }
}
"#,
        expect![[r#"ty: Foo, name: ?"#]],
    );
}

#[test]
fn expected_type_if_let_without_leading_char() {
    cov_mark::check!(expected_type_if_let_without_leading_char);
    check_expected_type_and_name(
        r#"
enum Foo { Bar, Baz, Quux }

fn foo() {
    let f = Foo::Quux;
    if let $0 = f { }
}
"#,
        expect![[r#"ty: Foo, name: ?"#]],
    )
}

#[test]
fn expected_type_if_let_with_leading_char() {
    cov_mark::check!(expected_type_if_let_with_leading_char);
    check_expected_type_and_name(
        r#"
enum Foo { Bar, Baz, Quux }

fn foo() {
    let f = Foo::Quux;
    if let c$0 = f { }
}
"#,
        expect![[r#"ty: Foo, name: ?"#]],
    )
}

#[test]
fn expected_type_fn_ret_without_leading_char() {
    cov_mark::check!(expected_type_fn_ret_without_leading_char);
    check_expected_type_and_name(
        r#"
fn foo() -> u32 {
    $0
}
"#,
        expect![[r#"ty: u32, name: ?"#]],
    )
}

#[test]
fn expected_type_fn_ret_with_leading_char() {
    cov_mark::check!(expected_type_fn_ret_with_leading_char);
    check_expected_type_and_name(
        r#"
fn foo() -> u32 {
    c$0
}
"#,
        expect![[r#"ty: u32, name: ?"#]],
    )
}

#[test]
fn expected_type_fn_ret_fn_ref_fully_typed() {
    check_expected_type_and_name(
        r#"
fn foo() -> u32 {
    foo$0
}
"#,
        expect![[r#"ty: u32, name: ?"#]],
    )
}

#[test]
fn expected_type_closure_param_return() {
    // FIXME: make this work with `|| $0`
    check_expected_type_and_name(
        r#"
//- minicore: fn
fn foo() {
    bar(|| a$0);
}

fn bar(f: impl FnOnce() -> u32) {}
"#,
        expect![[r#"ty: u32, name: ?"#]],
    );
}

#[test]
fn expected_type_generic_function() {
    check_expected_type_and_name(
        r#"
fn foo() {
    bar::<u32>($0);
}

fn bar<T>(t: T) {}
"#,
        expect![[r#"ty: u32, name: t"#]],
    );
}

#[test]
fn expected_type_generic_method() {
    check_expected_type_and_name(
        r#"
fn foo() {
    S(1u32).bar($0);
}

struct S<T>(T);
impl<T> S<T> {
    fn bar(self, t: T) {}
}
"#,
        expect![[r#"ty: u32, name: t"#]],
    );
}

#[test]
fn expected_type_functional_update() {
    cov_mark::check!(expected_type_struct_func_update);
    check_expected_type_and_name(
        r#"
struct Foo { field: u32 }
fn foo() {
    Foo {
        ..$0
    }
}
"#,
        expect![[r#"ty: Foo, name: ?"#]],
    );
}

#[test]
fn expected_type_param_pat() {
    check_expected_type_and_name(
        r#"
struct Foo { field: u32 }
fn foo(a$0: Foo) {}
"#,
        expect![[r#"ty: Foo, name: ?"#]],
    );
    check_expected_type_and_name(
        r#"
struct Foo { field: u32 }
fn foo($0: Foo) {}
"#,
        // FIXME make this work, currently fails due to pattern recovery eating the `:`
        expect![[r#"ty: ?, name: ?"#]],
    );
}

#[test]
fn expected_type_ref_prefix_on_field() {
    check_expected_type_and_name(
        r#"
fn foo(_: &mut i32) {}
struct S {
    field: i32,
}

fn main() {
    let s = S {
        field: 100,
    };
    foo(&mut s.f$0);
}
"#,
        expect!["ty: i32, name: ?"],
    );
}

#[test]
fn expected_type_ref_return_pos() {
    check_expected_type_and_name(
        r#"
fn f(thing: u32) -> &u32 {
    &thin$0
}
"#,
        expect!["ty: u32, name: ?"],
    );
}
