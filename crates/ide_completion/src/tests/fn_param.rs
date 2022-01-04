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
            kw mut
            sp Self
            st A
        "#]],
    )
}
