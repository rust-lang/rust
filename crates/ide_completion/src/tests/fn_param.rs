use expect_test::{expect, Expect};

use crate::tests::completion_list;

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual);
}

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
            kw ref
            kw mut
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
            kw ref
            kw mut
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
            kw ref
            kw mut
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
            kw ref
            kw mut
        "#]],
    );
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
            kw ref
            kw mut
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
            kw ref
            kw mut
        "#]],
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
            bn Bar { bar }: Bar
            kw ref
            kw mut
            bn Bar              Bar { bar$1 }: Bar$0
            st Bar
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
            bn self
            bn &self
            bn mut self
            bn &mut self
            bn file_id: usize
            kw ref
            kw mut
            sp Self
            st A
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
            bn file_id: usize
            kw ref
            kw mut
            sp Self
            st A
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
            bn foo: i32
            bn baz: i32
            bn bar: i32
            kw ref
            kw mut
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
            bn baz: i32
            bn bar: i32
            bn foo: i32
            kw ref
            kw mut
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
            kw ref
            kw mut
        "#]],
    )
}
