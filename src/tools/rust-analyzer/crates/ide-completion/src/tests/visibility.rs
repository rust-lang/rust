//! Completion tests for visibility modifiers.
use expect_test::{expect, Expect};

use crate::tests::{completion_list, completion_list_with_trigger_character};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual)
}

fn check_with_trigger_character(ra_fixture: &str, trigger_character: char, expect: Expect) {
    let actual = completion_list_with_trigger_character(ra_fixture, Some(trigger_character));
    expect.assert_eq(&actual)
}

#[test]
fn empty_pub() {
    cov_mark::check!(kw_completion_in);
    check_with_trigger_character(
        r#"
pub($0)
"#,
        '(',
        expect![[r#"
            kw crate
            kw in
            kw self
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
            kw crate
            kw self
        "#]],
    );
}

#[test]
fn qualified() {
    cov_mark::check!(visibility_qualified);
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
        pub(in crate::$0)
    }
    mod baz {}
}

mod bar {}
"#,
        expect![[r#"
            md qux
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
