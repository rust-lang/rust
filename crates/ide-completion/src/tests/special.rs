use expect_test::{expect, Expect};

use crate::tests::{check_edit, completion_list_no_kw};

fn check(ra_fixture: &str, expect: Expect) {
    let actual = completion_list_no_kw(ra_fixture);
    expect.assert_eq(&actual)
}

#[test]
fn completes_if_prefix_is_keyword() {
    check_edit(
        "wherewolf",
        r#"
fn main() {
    let wherewolf = 92;
    drop(where$0)
}
"#,
        r#"
fn main() {
    let wherewolf = 92;
    drop(wherewolf)
}
"#,
    )
}

/// Regression test for issue #6091.
#[test]
fn correctly_completes_module_items_prefixed_with_underscore() {
    check_edit(
        "_alpha",
        r#"
fn main() {
    _$0
}
fn _alpha() {}
"#,
        r#"
fn main() {
    _alpha()$0
}
fn _alpha() {}
"#,
    )
}

#[test]
fn completes_prelude() {
    check(
        r#"
//- /main.rs crate:main deps:std
fn foo() { let x: $0 }

//- /std/lib.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub struct Option;
    }
}
"#,
        expect![[r#"
                md std
                st Option
                bt u32
            "#]],
    );
}

#[test]
fn completes_prelude_macros() {
    check(
        r#"
//- /main.rs crate:main deps:std
fn f() {$0}

//- /std/lib.rs crate:std
pub mod prelude {
    pub mod rust_2018 {
        pub use crate::concat;
    }
}

mod macros {
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat { }
}
"#,
        expect![[r#"
                fn f()        fn()
                ma concat!(…) macro_rules! concat
                md std
                bt u32
            "#]],
    );
}

#[test]
fn completes_std_prelude_if_core_is_defined() {
    check(
        r#"
//- /main.rs crate:main deps:core,std
fn foo() { let x: $0 }

//- /core/lib.rs crate:core
pub mod prelude {
    pub mod rust_2018 {
        pub struct Option;
    }
}

//- /std/lib.rs crate:std deps:core
pub mod prelude {
    pub mod rust_2018 {
        pub struct String;
    }
}
"#,
        expect![[r#"
                md core
                md std
                st String
                bt u32
            "#]],
    );
}

#[test]
fn respects_doc_hidden() {
    check(
        r#"
//- /lib.rs crate:lib deps:std
fn f() {
    format_$0
}

//- /std.rs crate:std
#[doc(hidden)]
#[macro_export]
macro_rules! format_args_nl {
    () => {}
}

pub mod prelude {
    pub mod rust_2018 {}
}
            "#,
        expect![[r#"
                fn f() fn()
                md std
                bt u32
            "#]],
    );
}

#[test]
fn respects_doc_hidden_in_assoc_item_list() {
    check(
        r#"
//- /lib.rs crate:lib deps:std
struct S;
impl S {
    format_$0
}

//- /std.rs crate:std
#[doc(hidden)]
#[macro_export]
macro_rules! format_args_nl {
    () => {}
}

pub mod prelude {
    pub mod rust_2018 {}
}
            "#,
        expect![[r#"
                md std
            "#]],
    );
}
