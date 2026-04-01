//! Completion tests for visibility modifiers.
use expect_test::expect;

use crate::tests::{check, check_with_private_editable, check_with_trigger_character};

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

#[test]
fn use_inner_public_function() {
    check(
        r#"
//- /inner.rs crate:inner
pub fn inner_public() {}
fn inner_private() {}
//- /foo.rs crate:foo deps:inner
use inner::inner_public;
pub fn outer_public() {}
//- /lib.rs crate:lib deps:foo
fn x() {
    foo::$0
}
        "#,
        expect![[r#"
            fn outer_public() fn()
        "#]],
    );
}

#[test]
fn pub_use_inner_public_function() {
    check(
        r#"
//- /inner.rs crate:inner
pub fn inner_public() {}
fn inner_private() {}
//- /foo.rs crate:foo deps:inner
pub use inner::inner_public;
pub fn outer_public() {}
//- /lib.rs crate:lib deps:foo
fn x() {
    foo::$0
}
        "#,
        expect![[r#"
            fn inner_public() fn()
            fn outer_public() fn()
        "#]],
    );
}

#[test]
fn use_inner_public_function_private_editable() {
    check_with_private_editable(
        r#"
//- /inner.rs crate:inner
pub fn inner_public() {}
fn inner_private() {}
//- /foo.rs crate:foo deps:inner
use inner::inner_public;
pub fn outer_public() {}
//- /lib.rs crate:lib deps:foo
fn x() {
    foo::$0
}
        "#,
        expect![[r#"
            fn inner_public() fn()
            fn outer_public() fn()
        "#]],
    );
}

#[test]
fn pub_use_inner_public_function_private_editable() {
    check_with_private_editable(
        r#"
//- /inner.rs crate:inner
pub fn inner_public() {}
fn inner_private() {}
//- /foo.rs crate:foo deps:inner
pub use inner::inner_public;
pub fn outer_public() {}
//- /lib.rs crate:lib deps:foo
fn x() {
    foo::$0
}
        "#,
        expect![[r#"
            fn inner_public() fn()
            fn outer_public() fn()
        "#]],
    );
}
