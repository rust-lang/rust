use expect_test::{expect, Expect};

use crate::{
    context::{CompletionAnalysis, NameContext, NameKind, NameRefKind},
    tests::{check_edit, check_edit_with_config, TEST_CONFIG},
};

fn check(ra_fixture: &str, expect: Expect) {
    let config = TEST_CONFIG;
    let (db, position) = crate::tests::position(ra_fixture);
    let (ctx, analysis) = crate::context::CompletionContext::new(&db, position, &config).unwrap();

    let mut acc = crate::completions::Completions::default();
    if let CompletionAnalysis::Name(NameContext { kind: NameKind::IdentPat(pat_ctx), .. }) =
        &analysis
    {
        crate::completions::flyimport::import_on_the_fly_pat(&mut acc, &ctx, pat_ctx);
    }
    if let CompletionAnalysis::NameRef(name_ref_ctx) = &analysis {
        match &name_ref_ctx.kind {
            NameRefKind::Path(path) => {
                crate::completions::flyimport::import_on_the_fly_path(&mut acc, &ctx, path);
            }
            NameRefKind::DotAccess(dot_access) => {
                crate::completions::flyimport::import_on_the_fly_dot(&mut acc, &ctx, dot_access);
            }
            NameRefKind::Pattern(pattern) => {
                crate::completions::flyimport::import_on_the_fly_pat(&mut acc, &ctx, pattern);
            }
            _ => (),
        }
    }

    expect.assert_eq(&super::render_completion_list(Vec::from(acc)));
}

#[test]
fn function_fuzzy_completion() {
    check_edit(
        "stdin",
        r#"
//- /lib.rs crate:dep
pub mod io {
    pub fn stdin() {}
};

//- /main.rs crate:main deps:dep
fn main() {
    stdi$0
}
"#,
        r#"
use dep::io::stdin;

fn main() {
    stdin()$0
}
"#,
    );
}

#[test]
fn macro_fuzzy_completion() {
    check_edit(
        "macro_with_curlies!",
        r#"
//- /lib.rs crate:dep
/// Please call me as macro_with_curlies! {}
#[macro_export]
macro_rules! macro_with_curlies {
    () => {}
}

//- /main.rs crate:main deps:dep
fn main() {
    curli$0
}
"#,
        r#"
use dep::macro_with_curlies;

fn main() {
    macro_with_curlies! {$0}
}
"#,
    );
}

#[test]
fn struct_fuzzy_completion() {
    check_edit(
        "ThirdStruct",
        r#"
//- /lib.rs crate:dep
pub struct FirstStruct;
pub mod some_module {
    pub struct SecondStruct;
    pub struct ThirdStruct;
}

//- /main.rs crate:main deps:dep
use dep::{FirstStruct, some_module::SecondStruct};

fn main() {
    this$0
}
"#,
        r#"
use dep::{FirstStruct, some_module::{SecondStruct, ThirdStruct}};

fn main() {
    ThirdStruct
}
"#,
    );
}

#[test]
fn short_paths_are_ignored() {
    cov_mark::check!(flyimport_exact_on_short_path);

    check(
        r#"
//- /lib.rs crate:dep
pub struct Bar;
pub struct Rcar;
pub struct Rc;
pub mod some_module {
    pub struct Bar;
    pub struct Rcar;
    pub struct Rc;
}

//- /main.rs crate:main deps:dep
fn main() {
    rc$0
}
"#,
        expect![[r#"
            st Rc (use dep::Rc)
            st Rc (use dep::some_module::Rc)
        "#]],
    );
}

#[test]
fn fuzzy_completions_come_in_specific_order() {
    cov_mark::check!(certain_fuzzy_order_test);
    check(
        r#"
//- /lib.rs crate:dep
pub struct FirstStruct;
pub mod some_module {
    // already imported, omitted
    pub struct SecondStruct;
    // does not contain all letters from the query, omitted
    pub struct UnrelatedOne;
    // contains all letters from the query, but not in sequence, displayed last
    pub struct ThiiiiiirdStruct;
    // contains all letters from the query, but not in the beginning, displayed second
    pub struct AfterThirdStruct;
    // contains all letters from the query in the beginning, displayed first
    pub struct ThirdStruct;
}

//- /main.rs crate:main deps:dep
use dep::{FirstStruct, some_module::SecondStruct};

fn main() {
    hir$0
}
"#,
        expect![[r#"
                st ThirdStruct (use dep::some_module::ThirdStruct)
                st AfterThirdStruct (use dep::some_module::AfterThirdStruct)
                st ThiiiiiirdStruct (use dep::some_module::ThiiiiiirdStruct)
            "#]],
    );
}

#[test]
fn trait_function_fuzzy_completion() {
    let fixture = r#"
        //- /lib.rs crate:dep
        pub mod test_mod {
            pub trait TestTrait {
                const SPECIAL_CONST: u8;
                type HumbleType;
                fn weird_function();
                fn random_method(&self);
            }
            pub struct TestStruct {}
            impl TestTrait for TestStruct {
                const SPECIAL_CONST: u8 = 42;
                type HumbleType = ();
                fn weird_function() {}
                fn random_method(&self) {}
            }
        }

        //- /main.rs crate:main deps:dep
        fn main() {
            dep::test_mod::TestStruct::wei$0
        }
        "#;

    check(
        fixture,
        expect![[r#"
                fn weird_function() (use dep::test_mod::TestTrait) fn()
            "#]],
    );

    check_edit(
        "weird_function",
        fixture,
        r#"
use dep::test_mod::TestTrait;

fn main() {
    dep::test_mod::TestStruct::weird_function()$0
}
"#,
    );
}

#[test]
fn trait_const_fuzzy_completion() {
    let fixture = r#"
        //- /lib.rs crate:dep
        pub mod test_mod {
            pub trait TestTrait {
                const SPECIAL_CONST: u8;
                type HumbleType;
                fn weird_function();
                fn random_method(&self);
            }
            pub struct TestStruct {}
            impl TestTrait for TestStruct {
                const SPECIAL_CONST: u8 = 42;
                type HumbleType = ();
                fn weird_function() {}
                fn random_method(&self) {}
            }
        }

        //- /main.rs crate:main deps:dep
        fn main() {
            dep::test_mod::TestStruct::spe$0
        }
        "#;

    check(
        fixture,
        expect![[r#"
            ct SPECIAL_CONST (use dep::test_mod::TestTrait)
        "#]],
    );

    check_edit(
        "SPECIAL_CONST",
        fixture,
        r#"
use dep::test_mod::TestTrait;

fn main() {
    dep::test_mod::TestStruct::SPECIAL_CONST
}
"#,
    );
}

#[test]
fn trait_method_fuzzy_completion() {
    let fixture = r#"
        //- /lib.rs crate:dep
        pub mod test_mod {
            pub trait TestTrait {
                const SPECIAL_CONST: u8;
                type HumbleType;
                fn weird_function();
                fn random_method(&self);
            }
            pub struct TestStruct {}
            impl TestTrait for TestStruct {
                const SPECIAL_CONST: u8 = 42;
                type HumbleType = ();
                fn weird_function() {}
                fn random_method(&self) {}
            }
        }

        //- /main.rs crate:main deps:dep
        fn main() {
            let test_struct = dep::test_mod::TestStruct {};
            test_struct.ran$0
        }
        "#;

    check(
        fixture,
        expect![[r#"
                me random_method() (use dep::test_mod::TestTrait) fn(&self)
            "#]],
    );

    check_edit(
        "random_method",
        fixture,
        r#"
use dep::test_mod::TestTrait;

fn main() {
    let test_struct = dep::test_mod::TestStruct {};
    test_struct.random_method()$0
}
"#,
    );
}

#[test]
fn trait_method_from_alias() {
    let fixture = r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub trait TestTrait {
        fn random_method();
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn random_method() {}
    }
    pub type TestAlias = TestStruct;
}

//- /main.rs crate:main deps:dep
fn main() {
    dep::test_mod::TestAlias::ran$0
}
"#;

    check(
        fixture,
        expect![[r#"
                fn random_method() (use dep::test_mod::TestTrait) fn()
            "#]],
    );

    check_edit(
        "random_method",
        fixture,
        r#"
use dep::test_mod::TestTrait;

fn main() {
    dep::test_mod::TestAlias::random_method()$0
}
"#,
    );
}

#[test]
fn no_trait_type_fuzzy_completion() {
    check(
        r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
fn main() {
    dep::test_mod::TestStruct::hum$0
}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn does_not_propose_names_in_scope() {
    check(
        r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
use dep::test_mod::TestStruct;
fn main() {
    TestSt$0
}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn does_not_propose_traits_in_scope() {
    check(
        r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
use dep::test_mod::{TestStruct, TestTrait};
fn main() {
    dep::test_mod::TestStruct::hum$0
}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn blanket_trait_impl_import() {
    check_edit(
        "another_function",
        r#"
//- /lib.rs crate:dep
pub mod test_mod {
    pub struct TestStruct {}
    pub trait TestTrait {
        fn another_function();
    }
    impl<T> TestTrait for T {
        fn another_function() {}
    }
}

//- /main.rs crate:main deps:dep
fn main() {
    dep::test_mod::TestStruct::ano$0
}
"#,
        r#"
use dep::test_mod::TestTrait;

fn main() {
    dep::test_mod::TestStruct::another_function()$0
}
"#,
    );
}

#[test]
fn zero_input_deprecated_assoc_item_completion() {
    check(
        r#"
//- /lib.rs crate:dep
pub mod test_mod {
    #[deprecated]
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
fn main() {
    let test_struct = dep::test_mod::TestStruct {};
    test_struct.$0
}
        "#,
        expect![[r#"
                me random_method() (use dep::test_mod::TestTrait) fn(&self) DEPRECATED
            "#]],
    );

    check(
        r#"
//- /lib.rs crate:dep
pub mod test_mod {
    #[deprecated]
    pub trait TestTrait {
        const SPECIAL_CONST: u8;
        type HumbleType;
        fn weird_function();
        fn random_method(&self);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        const SPECIAL_CONST: u8 = 42;
        type HumbleType = ();
        fn weird_function() {}
        fn random_method(&self) {}
    }
}

//- /main.rs crate:main deps:dep
fn main() {
    dep::test_mod::TestStruct::$0
}
"#,
        expect![[r#"
            ct SPECIAL_CONST (use dep::test_mod::TestTrait) DEPRECATED
            fn weird_function() (use dep::test_mod::TestTrait) fn() DEPRECATED
        "#]],
    );
}

#[test]
fn no_completions_in_use_statements() {
    check(
        r#"
//- /lib.rs crate:dep
pub mod io {
    pub fn stdin() {}
};

//- /main.rs crate:main deps:dep
use stdi$0

fn main() {}
"#,
        expect![[]],
    );
}

#[test]
fn prefix_config_usage() {
    let fixture = r#"
mod foo {
    pub mod bar {
        pub struct Item;
    }
}

use crate::foo::bar;

fn main() {
    Ite$0
}"#;
    let mut config = TEST_CONFIG;

    config.insert_use.prefix_kind = hir::PrefixKind::ByCrate;
    check_edit_with_config(
        config.clone(),
        "Item",
        fixture,
        r#"
mod foo {
    pub mod bar {
        pub struct Item;
    }
}

use crate::foo::bar::{self, Item};

fn main() {
    Item
}"#,
    );

    config.insert_use.prefix_kind = hir::PrefixKind::BySelf;
    check_edit_with_config(
        config.clone(),
        "Item",
        fixture,
        r#"
mod foo {
    pub mod bar {
        pub struct Item;
    }
}

use crate::foo::bar;

use self::foo::bar::Item;

fn main() {
    Item
}"#,
    );

    config.insert_use.prefix_kind = hir::PrefixKind::Plain;
    check_edit_with_config(
        config,
        "Item",
        fixture,
        r#"
mod foo {
    pub mod bar {
        pub struct Item;
    }
}

use foo::bar::Item;

use crate::foo::bar;

fn main() {
    Item
}"#,
    );
}

#[test]
fn unresolved_qualifier() {
    let fixture = r#"
mod foo {
    pub mod bar {
        pub mod baz {
            pub struct Item;
        }
    }
}

fn main() {
    bar::baz::Ite$0
}"#;

    check(
        fixture,
        expect![[r#"
        st Item (use foo::bar::baz::Item)
        "#]],
    );

    check_edit(
        "Item",
        fixture,
        r#"
        use foo::bar;

        mod foo {
            pub mod bar {
                pub mod baz {
                    pub struct Item;
                }
            }
        }

        fn main() {
            bar::baz::Item
        }"#,
    );
}

#[test]
fn unresolved_assoc_item_container() {
    let fixture = r#"
mod foo {
    pub struct Item;

    impl Item {
        pub const TEST_ASSOC: usize = 3;
    }
}

fn main() {
    Item::TEST_A$0
}"#;

    check(
        fixture,
        expect![[r#"
        ct TEST_ASSOC (use foo::Item)
        "#]],
    );

    check_edit(
        "TEST_ASSOC",
        fixture,
        r#"
use foo::Item;

mod foo {
    pub struct Item;

    impl Item {
        pub const TEST_ASSOC: usize = 3;
    }
}

fn main() {
    Item::TEST_ASSOC
}"#,
    );
}

#[test]
fn unresolved_assoc_item_container_with_path() {
    let fixture = r#"
mod foo {
    pub mod bar {
        pub struct Item;

        impl Item {
            pub const TEST_ASSOC: usize = 3;
        }
    }
}

fn main() {
    bar::Item::TEST_A$0
}"#;

    check(
        fixture,
        expect![[r#"
        ct TEST_ASSOC (use foo::bar::Item)
    "#]],
    );

    check_edit(
        "TEST_ASSOC",
        fixture,
        r#"
use foo::bar;

mod foo {
    pub mod bar {
        pub struct Item;

        impl Item {
            pub const TEST_ASSOC: usize = 3;
        }
    }
}

fn main() {
    bar::Item::TEST_ASSOC
}"#,
    );
}

#[test]
fn fuzzy_unresolved_path() {
    check(
        r#"
mod foo {
    pub mod bar {
        pub struct Item;

        impl Item {
            pub const TEST_ASSOC: usize = 3;
        }
    }
}

fn main() {
    bar::ASS$0
}"#,
        expect![[]],
    )
}

#[test]
fn unqualified_assoc_items_are_omitted() {
    check(
        r#"
mod something {
    pub trait BaseTrait {
        fn test_function() -> i32;
    }

    pub struct Item1;
    pub struct Item2;

    impl BaseTrait for Item1 {
        fn test_function() -> i32 {
            1
        }
    }

    impl BaseTrait for Item2 {
        fn test_function() -> i32 {
            2
        }
    }
}

fn main() {
    test_f$0
}"#,
        expect![[]],
    )
}

#[test]
fn case_matters() {
    check(
        r#"
mod foo {
    pub const TEST_CONST: usize = 3;
    pub fn test_function() -> i32 {
        4
    }
}

fn main() {
    TES$0
}"#,
        expect![[r#"
        ct TEST_CONST (use foo::TEST_CONST)
    "#]],
    );

    check(
        r#"
mod foo {
    pub const TEST_CONST: usize = 3;
    pub fn test_function() -> i32 {
        4
    }
}

fn main() {
    tes$0
}"#,
        expect![[r#"
        ct TEST_CONST (use foo::TEST_CONST)
        fn test_function() (use foo::test_function) fn() -> i32
    "#]],
    );

    check(
        r#"
mod foo {
    pub const TEST_CONST: usize = 3;
    pub fn test_function() -> i32 {
        4
    }
}

fn main() {
    Te$0
}"#,
        expect![[]],
    );
}

#[test]
fn no_fuzzy_during_fields_of_record_lit_syntax() {
    check(
        r#"
mod m {
    pub fn some_fn() -> i32 {
        42
    }
}
struct Foo {
    some_field: i32,
}
fn main() {
    let _ = Foo { so$0 };
}
"#,
        expect![[]],
    );
}

#[test]
fn fuzzy_after_fields_of_record_lit_syntax() {
    check(
        r#"
mod m {
    pub fn some_fn() -> i32 {
        42
    }
}
struct Foo {
    some_field: i32,
}
fn main() {
    let _ = Foo { some_field: som$0 };
}
"#,
        expect![[r#"
                fn some_fn() (use m::some_fn) fn() -> i32
            "#]],
    );
}

#[test]
fn no_flyimports_in_traits_and_impl_declarations() {
    check(
        r#"
mod m {
    pub fn some_fn() -> i32 {
        42
    }
}
trait Foo {
    som$0
}
"#,
        expect![[r#""#]],
    );

    check(
        r#"
mod m {
    pub fn some_fn() -> i32 {
        42
    }
}
struct Foo;
impl Foo {
    som$0
}
"#,
        expect![[r#""#]],
    );

    check(
        r#"
mod m {
    pub fn some_fn() -> i32 {
        42
    }
}
struct Foo;
trait Bar {}
impl Bar for Foo {
    som$0
}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn no_inherent_candidates_proposed() {
    check(
        r#"
mod baz {
    pub trait DefDatabase {
        fn method1(&self);
    }
    pub trait HirDatabase: DefDatabase {
        fn method2(&self);
    }
}

mod bar {
    fn test(db: &dyn crate::baz::HirDatabase) {
        db.metho$0
    }
}
            "#,
        expect![[r#""#]],
    );
    check(
        r#"
mod baz {
    pub trait DefDatabase {
        fn method1(&self);
    }
    pub trait HirDatabase: DefDatabase {
        fn method2(&self);
    }
}

mod bar {
    fn test(db: &impl crate::baz::HirDatabase) {
        db.metho$0
    }
}
"#,
        expect![[r#""#]],
    );
    check(
        r#"
mod baz {
    pub trait DefDatabase {
        fn method1(&self);
    }
    pub trait HirDatabase: DefDatabase {
        fn method2(&self);
    }
}

mod bar {
    fn test<T: crate::baz::HirDatabase>(db: T) {
        db.metho$0
    }
}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn respects_doc_hidden() {
    check(
        r#"
//- /lib.rs crate:lib deps:dep
fn f() {
    ().fro$0
}

//- /dep.rs crate:dep
#[doc(hidden)]
pub trait Private {
    fn frob(&self) {}
}

impl<T> Private for T {}
            "#,
        expect![[r#""#]],
    );
    check(
        r#"
//- /lib.rs crate:lib deps:dep
fn f() {
    ().fro$0
}

//- /dep.rs crate:dep
pub trait Private {
    #[doc(hidden)]
    fn frob(&self) {}
}

impl<T> Private for T {}
            "#,
        expect![[r#""#]],
    );
}

#[test]
fn regression_9760() {
    check(
        r#"
struct Struct;
fn main() {}

mod mud {
    fn func() {
        let struct_instance = Stru$0
    }
}
"#,
        expect![[r#"
                st Struct (use crate::Struct)
            "#]],
    );
}

#[test]
fn flyimport_pattern() {
    check(
        r#"
mod module {
    pub struct FooStruct {}
    pub const FooConst: () = ();
    pub fn foo_fun() {}
}
fn function() {
    let foo$0
}
"#,
        expect![[r#"
            ct FooConst (use module::FooConst)
            st FooStruct (use module::FooStruct)
        "#]],
    );
}

#[test]
fn flyimport_pattern_no_unstable_item_on_stable() {
    check(
        r#"
//- /main.rs crate:main deps:std
fn function() {
    let foo$0
}
//- /std.rs crate:std
#[unstable]
pub struct FooStruct {}
"#,
        expect![""],
    );
}

#[test]
fn flyimport_pattern_unstable_item_on_nightly() {
    check(
        r#"
//- toolchain:nightly
//- /main.rs crate:main deps:std
fn function() {
    let foo$0
}
//- /std.rs crate:std
#[unstable]
pub struct FooStruct {}
"#,
        expect![[r#"
            st FooStruct (use std::FooStruct)
        "#]],
    );
}

#[test]
fn flyimport_item_name() {
    check(
        r#"
mod module {
    pub struct Struct;
}
struct Str$0
    "#,
        expect![[r#""#]],
    );
}

#[test]
fn flyimport_rename() {
    check(
        r#"
mod module {
    pub struct Struct;
}
use self as Str$0;
    "#,
        expect![[r#""#]],
    );
}

#[test]
fn flyimport_enum_variant() {
    check(
        r#"
mod foo {
    pub struct Barbara;
}

enum Foo {
    Barba$0()
}
}"#,
        expect![[r#""#]],
    );

    check(
        r#"
mod foo {
    pub struct Barbara;
}

enum Foo {
    Barba(Barba$0)
}
}"#,
        expect![[r#"
            st Barbara (use foo::Barbara)
        "#]],
    )
}

#[test]
fn flyimport_attribute() {
    check(
        r#"
//- proc_macros:identity
#[ide$0]
struct Foo;
"#,
        expect![[r#"
            at identity (use proc_macros::identity) proc_macro identity
        "#]],
    );
    check_edit(
        "identity",
        r#"
//- proc_macros:identity
#[ide$0]
struct Foo;
"#,
        r#"
use proc_macros::identity;

#[identity]
struct Foo;
"#,
    );
}

#[test]
fn flyimport_in_type_bound_omits_types() {
    check(
        r#"
mod module {
    pub struct CompletemeStruct;
    pub type CompletemeType = ();
    pub enum CompletemeEnum {}
    pub trait CompletemeTrait {}
}

fn f<T>() where T: Comp$0
"#,
        expect![[r#"
            tt CompletemeTrait (use module::CompletemeTrait)
        "#]],
    );
}

#[test]
fn flyimport_source_file() {
    check(
        r#"
//- /main.rs crate:main deps:dep
def$0
//- /lib.rs crate:dep
#[macro_export]
macro_rules! define_struct {
    () => {
        pub struct Foo;
    };
}
"#,
        expect![[r#"
            ma define_struct!(…) (use dep::define_struct) macro_rules! define_struct
        "#]],
    );
}

#[test]
fn macro_use_prelude_is_in_scope() {
    check(
        r#"
//- /main.rs crate:main deps:dep
#[macro_use]
extern crate dep;

fn main() {
    print$0
}
//- /lib.rs crate:dep
#[macro_export]
macro_rules! println {
    () => {}
}
"#,
        expect![""],
    )
}
