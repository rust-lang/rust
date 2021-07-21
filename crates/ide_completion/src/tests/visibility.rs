//! Completion tests for visibility modifiers.
use expect_test::{expect, Expect};

use crate::tests::completion_list;

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual)
}

#[test]
fn empty_pub() {
    cov_mark::check!(kw_completion_in);
    check(
        r#"
pub($0)
"#,
        expect![[r#"
            kw in
            kw self
            kw super
            kw crate
        "#]],
    );
}

#[test]
fn after_in_kw() {
    check(
        r#"
pub(in $0)
"#,
        expect![[r#"
            kw self
            kw super
            kw crate
        "#]],
    );
}

#[test]
fn qualified() {
    check(
        r#"
mod foo {
    pub(in crate::$0)
}

mod bar {}
"#,
        expect![[r#"
            md foo
        "#]],
    );
    check(
        r#"
mod qux {
    mod foo {
        pub(in crate::qux::$0)
    }
    mod baz {}
}

mod bar {}
"#,
        expect![[r#"
            md foo
        "#]],
    );
}
