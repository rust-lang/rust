use expect_test::expect;

use crate::tests::{check, check_with_trigger_character};

#[test]
fn only_param() {
    check(
        r#"
fn foo(file_id: usize) {}
fn bar(file_id: usize) {}
fn baz(file$0) {}
"#,
        expect![[r#"
            bn file_id: usize
            kw mut
            kw ref
        "#]],
    );
}

#[test]
fn last_param() {
    check(
        r#"
fn foo(file_id: usize) {}
fn bar(file_id: usize) {}
fn baz(foo: (), file$0) {}
"#,
        expect![[r#"
            bn file_id: usize
            kw mut
            kw ref
        "#]],
    );
}

#[test]
fn first_param() {
    check(
        r#"
fn foo(file_id: usize) {}
fn bar(file_id: usize) {}
fn baz(file$0 id: u32) {}
"#,
        expect![[r#"
            bn file_id: usize,
            kw mut
            kw ref
        "#]],
    );
}

#[test]
fn repeated_param_name() {
    check(
        r#"
fn foo(file_id: usize) {}
fn bar(file_id: u32, $0) {}
"#,
        expect![[r#"
            kw mut
            kw ref
        "#]],
    );

    check(
        r#"
fn f(#[foo = "bar"] baz: u32,) {}
fn g(baz: (), ba$0)
"#,
        expect![[r#"
            kw mut
            kw ref
        "#]],
    )
}

#[test]
fn trait_param() {
    check(
        r#"
pub(crate) trait SourceRoot {
    pub fn contains(file_id: usize) -> bool;
    pub fn syntax(file$0)
}
"#,
        expect![[r#"
            bn file_id: usize
            kw mut
            kw ref
        "#]],
    );
}

#[test]
fn in_inner_function() {
    check(
        r#"
fn outer(text: &str) {
    fn inner($0)
}
"#,
        expect![[r#"
            bn text: &str
            kw mut
            kw ref
        "#]],
    )
}

#[test]
fn trigger_by_l_paren() {
    check_with_trigger_character(
        r#"
fn foo($0)
"#,
        Some('('),
        expect![[]],
    )
}

#[test]
fn shows_non_ident_pat_param() {
    check(
        r#"
struct Bar { bar: u32 }
fn foo(Bar { bar }: Bar) {}
fn foo2($0) {}
"#,
        expect![[r#"
            st Bar
            bn Bar { bar }: Bar
            bn Bar {…} Bar { bar$1 }: Bar$0
            kw mut
            kw ref
        "#]],
    )
}

#[test]
fn in_impl_only_param() {
    check(
        r#"
struct A {}

impl A {
    fn foo(file_id: usize) {}
    fn new($0) {}
}
"#,
        expect![[r#"
            sp Self
            st A
            bn &mut self
            bn &self
            bn file_id: usize
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    )
}

#[test]
fn in_impl_after_self() {
    check(
        r#"
struct A {}

impl A {
    fn foo(file_id: usize) {}
    fn new(self, $0) {}
}
"#,
        expect![[r#"
            sp Self
            st A
            bn file_id: usize
            kw mut
            kw ref
        "#]],
    )
}

// doesn't complete qux due to there being no expression after
// see source_analyzer::adjust comment
#[test]
fn local_fn_shows_locals_for_params() {
    check(
        r#"
fn outer() {
    let foo = 3;
    {
        let bar = 3;
        fn inner($0) {}
        let baz = 3;
        let qux = 3;
    }
    let fez = 3;
}
"#,
        expect![[r#"
            bn bar: i32
            bn baz: i32
            bn foo: i32
            kw mut
            kw ref
        "#]],
    )
}

#[test]
fn closure_shows_locals_for_params() {
    check(
        r#"
fn outer() {
    let foo = 3;
    {
        let bar = 3;
        |$0| {};
        let baz = 3;
        let qux = 3;
    }
    let fez = 3;
}
"#,
        expect![[r#"
            bn bar: i32
            bn baz: i32
            bn foo: i32
            kw mut
            kw ref
        "#]],
    )
}

#[test]
fn completes_fully_equal() {
    check(
        r#"
fn foo(bar: u32) {}
fn bar(bar$0) {}
"#,
        expect![[r#"
            bn bar: u32
            kw mut
            kw ref
        "#]],
    )
}

#[test]
fn completes_for_params_with_attributes() {
    check(
        r#"
fn f(foo: (), #[baz = "qux"] mut bar: u32) {}
fn g(foo: (), #[baz = "qux"] mut ba$0)
"#,
        expect![[r##"
            bn #[baz = "qux"] mut bar: u32
        "##]],
    )
}
