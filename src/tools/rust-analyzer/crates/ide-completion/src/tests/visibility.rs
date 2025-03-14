//! Completion tests for visibility modifiers.
use expect_test::expect;

use crate::tests::{check, check_with_trigger_character};

#[test]
fn empty_pub() {
    cov_mark::check!(kw_completion_in);
    check_with_trigger_character(
        r#"
pub($0)
"#,
        Some('('),
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
