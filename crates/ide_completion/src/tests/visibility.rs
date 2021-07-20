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
