//! Completion tests for use trees.
use expect_test::{expect, Expect};

use crate::tests::completion_list;

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list(ra_fixture);
    expect.assert_eq(&actual)
}

#[test]
fn use_tree_start() {
    cov_mark::check!(unqualified_path_selected_only);
    check(
        r#"
//- /lib.rs crate:main deps:other_crate
use f$0

struct Foo;
enum FooBar {
    Foo,
    Bar
}
mod foo {}
//- /other_crate/lib.rs crate:other_crate
// nothing here
"#,
        expect![[r#"
            en FooBar::
            md foo
            md other_crate
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn use_tree_start_abs() {
    cov_mark::check!(use_tree_crate_roots_only);
    check(
        r#"
//- /lib.rs crate:main deps:other_crate
use ::f$0

struct Foo;
mod foo {}
//- /other_crate/lib.rs crate:other_crate
// nothing here
"#,
        expect![[r#"
            md other_crate
        "#]],
    );
}

#[test]
fn dont_complete_current_use() {
    cov_mark::check!(dont_complete_current_use);
    check(r#"use self::foo$0;"#, expect![[r#""#]]);
    check(
        r#"
mod foo { pub struct S; }
use self::{foo::*, bar$0};
"#,
        expect![[r#"
            md foo
            st S
        "#]],
    );
}

#[test]
fn nested_use_tree() {
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
            md bar
            kw self
        "#]],
    );
}

#[test]
fn deeply_nested_use_tree() {
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
            md baz
            kw self
        "#]],
    );
}

#[test]
fn plain_qualified_use_tree() {
    check(
        r#"
use foo::$0

mod foo {
    struct Private;
    pub struct Foo;
    macro_rules! foo_ { {} => {} }
    pub use foo_ as foo;
}
struct Bar;
"#,
        expect![[r#"
            ma foo macro_rules! foo_
            st Foo
        "#]],
    );
}

#[test]
fn enum_plain_qualified_use_tree() {
    cov_mark::check!(enum_plain_qualified_use_tree);
    check(
        r#"
use Foo::$0

enum Foo {
    UnitVariant,
    TupleVariant(),
    RecordVariant {},
}
impl Foo {
    const CONST: () = ()
    fn func() {}
}
"#,
        expect![[r#"
            ev RecordVariant RecordVariant
            ev TupleVariant  TupleVariant
            ev UnitVariant   UnitVariant
        "#]],
    );
}

#[test]
fn self_qualified_use_tree() {
    check(
        r#"
use self::$0

mod foo {}
struct Bar;
"#,
        expect![[r#"
            md foo
            st Bar
        "#]],
    );
}

#[test]
fn super_qualified_use_tree() {
    check(
        r#"
mod bar {
    use super::$0
}

mod foo {}
struct Bar;
"#,
        expect![[r#"
            md bar
            md foo
            st Bar
        "#]],
    );
}

#[test]
fn super_super_qualified_use_tree() {
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
            ct A
            md b
            kw super::
        "#]],
    );
}

#[test]
fn crate_qualified_use_tree() {
    check(
        r#"
use crate::$0

mod foo {}
struct Bar;
"#,
        expect![[r#"
            md foo
            st Bar
        "#]],
    );
}

#[test]
fn extern_crate_qualified_use_tree() {
    check(
        r#"
//- /lib.rs crate:main deps:other_crate
use other_crate::$0
//- /other_crate/lib.rs crate:other_crate
pub struct Foo;
pub mod foo {}
"#,
        expect![[r#"
            md foo
            st Foo
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
            md bar
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn pub_suggest_use_tree_super_acc_to_depth_in_tree() {
    // https://github.com/rust-lang/rust-analyzer/issues/12439
    // Check discussion in https://github.com/rust-lang/rust-analyzer/pull/12447

    check(
        r#"
mod foo {
    mod bar {
        pub use super::$0;
    }
}
"#,
        expect![[r#"
            md bar
            kw super::
        "#]],
    );

    // Not suggest super when at crate root
    check(
        r#"
mod foo {
    mod bar {
        pub use super::super::$0;
    }
}
"#,
        expect![[r#"
            md foo
        "#]],
    );

    check(
        r#"
mod foo {
    use $0;
}
"#,
        expect![[r#"
            kw crate::
            kw self::
            kw super::
        "#]],
    );

    // Not suggest super after another kw in path ( here it is foo1 )
    check(
        r#"
mod foo {
    mod bar {
        use super::super::foo1::$0;
    }
}

mod foo1 {
    pub mod bar1 {}
}
"#,
        expect![[r#"
            md bar1
        "#]],
    );
}

#[test]
fn use_tree_braces_at_start() {
    check(
        r#"
struct X;
mod bar {}
use {$0};
"#,
        expect![[r#"
            md bar
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn impl_prefix_does_not_add_fn_snippet() {
    // regression test for 7222
    check(
        r#"
mod foo {
    pub fn bar(x: u32) {}
}
use self::foo::impl$0
"#,
        expect![[r#"
            fn bar fn(u32)
        "#]],
    );
}

#[test]
fn use_tree_no_unstable_items_on_stable() {
    check(
        r#"
//- /lib.rs crate:main deps:std
use std::$0
//- /std.rs crate:std
#[unstable]
pub mod simd {}
#[unstable]
pub struct S;
#[unstable]
pub fn foo() {}
#[unstable]
#[macro_export]
marco_rules! m { () => {} }
"#,
        expect![""],
    );
}

#[test]
fn use_tree_unstable_items_on_nightly() {
    check(
        r#"
//- toolchain:nightly
//- /lib.rs crate:main deps:std
use std::$0
//- /std.rs crate:std
#[unstable]
pub mod simd {}
#[unstable]
pub struct S;
#[unstable]
pub fn foo() {}
#[unstable]
#[macro_export]
marco_rules! m { () => {} }
"#,
        expect![[r#"
            fn foo  fn()
            md simd
            st S
        "#]],
    );
}
