use expect_test::{expect, Expect};

use crate::tests::completion_list;

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual)
}

#[test]
fn use_tree_start() {
    cov_mark::check!(only_completes_modules_in_import);
    check(
        r#"
//- /lib.rs crate:main deps:other_crate
use f$0

struct Foo;
mod foo {}
//- /other_crate/lib.rs crate:other_crate
// nothing here
"#,
        // FIXME: self in this case should also get the colons
        expect![[r#"
            kw crate::
            kw self
            kw super::
            md foo
            md other_crate
        "#]],
    );
}

#[test]
fn dont_complete_current_use() {
    cov_mark::check!(dont_complete_current_use);
    // FIXME: self shouldn't be here
    check(
        r#"use self::foo$0;"#,
        expect![[r#"
        kw self
    "#]],
    );
    check(
        r#"
mod foo { pub struct S; }
use self::{foo::*, bar$0};
"#,
        expect![[r#"
            kw self
            st S
            md foo
        "#]],
    );
}

#[test]
fn nested_use_tree() {
    // FIXME: self shouldn't be here
    check(
        r#"
mod foo {
    pub mod bar {
        pub struct FooBar;
    }
}
use foo::{bar::$0}
"#,
        expect![[r#"
            kw self
            st FooBar
        "#]],
    );
    check(
        r#"
mod foo {
    pub mod bar {
        pub struct FooBar;
    }
}
use foo::{$0}
"#,
        expect![[r#"
            kw self
            md bar
        "#]],
    );
}

#[test]
fn deeply_nested_use_tree() {
    // FIXME: self shouldn't be here
    check(
        r#"
mod foo {
    pub mod bar {
        pub mod baz {
            pub struct FooBarBaz;
        }
    }
}
use foo::{bar::{baz::$0}}
"#,
        expect![[r#"
            kw self
            st FooBarBaz
        "#]],
    );
    check(
        r#"
mod foo {
    pub mod bar {
        pub mod baz {
            pub struct FooBarBaz;
        }
    }
}
use foo::{bar::{$0}}
"#,
        expect![[r#"
            kw self
            md baz
        "#]],
    );
}

#[test]
fn plain_qualified_use_tree() {
    // FIXME: self shouldn't be here
    check(
        r#"
use foo::$0

mod foo {
    struct Private;
    pub struct Foo;
}
struct Bar;
"#,
        expect![[r#"
            kw self
            st Foo
        "#]],
    );
}

#[test]
fn self_qualified_use_tree() {
    // FIXME: self shouldn't be here
    check(
        r#"
use self::$0

mod foo {}
struct Bar;
"#,
        expect![[r#"
            kw self
            md foo
            st Bar
        "#]],
    );
}

#[test]
fn super_qualified_use_tree() {
    // FIXME: self shouldn't be here
    check(
        r#"
mod bar {
    use super::$0
}

mod foo {}
struct Bar;
"#,
        expect![[r#"
            kw self
            kw super::
            st Bar
            md bar
            md foo
        "#]],
    );
}

#[test]
fn super_super_qualified_use_tree() {
    // FIXME: self shouldn't be here
    check(
        r#"
mod a {
    const A: usize = 0;
    mod b {
        const B: usize = 0;
        mod c { use super::super::$0 }
    }
}
"#,
        expect![[r#"
            kw self
            kw super::
            md b
            ct A
        "#]],
    );
}

#[test]
fn crate_qualified_use_tree() {
    // FIXME: self shouldn't be here
    check(
        r#"
use crate::$0

mod foo {}
struct Bar;
"#,
        expect![[r#"
            kw self
            md foo
            st Bar
        "#]],
    );
}

#[test]
fn extern_crate_qualified_use_tree() {
    // FIXME: self shouldn't be here
    check(
        r#"
//- /lib.rs crate:main deps:other_crate
use other_crate::$0
//- /other_crate/lib.rs crate:other_crate
pub struct Foo;
pub mod foo {}
"#,
        expect![[r#"
            kw self
            st Foo
            md foo
        "#]],
    );
}

#[test]
fn pub_use_tree() {
    check(
        r#"
pub struct X;
pub mod bar {}
pub use $0;
"#,
        expect![[r#"
            kw crate::
            kw self
            kw super::
            md bar
        "#]],
    );
}
