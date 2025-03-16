use expect_test::{Expect, expect};

use crate::{
    CompletionConfig,
    context::{CompletionAnalysis, NameContext, NameKind, NameRefKind},
    tests::{TEST_CONFIG, check_edit, check_edit_with_config},
};

fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    check_with_config(TEST_CONFIG, ra_fixture, expect);
}

fn check_with_config(
    config: CompletionConfig<'_>,
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    expect: Expect,
) {
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
    stdin();$0
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
use dep::{some_module::{SecondStruct, ThirdStruct}, FirstStruct};

fn main() {
    ThirdStruct
}
"#,
    );
}

#[test]
fn short_paths_are_prefix_matched() {
    cov_mark::check!(flyimport_prefix_on_short_path);

    check(
        r#"
//- /lib.rs crate:dep
pub struct Barc;
pub struct Rcar;
pub struct Rc;
pub const RC: () = ();
pub mod some_module {
    pub struct Bar;
    pub struct Rcar;
    pub struct Rc;
    pub const RC: () = ();
}

//- /main.rs crate:main deps:dep
fn main() {
    Rc$0
}
"#,
        expect![[r#"
            st Rc (use dep::Rc)                    Rc
            st Rcar (use dep::Rcar)              Rcar
            st Rc (use dep::some_module::Rc)       Rc
            st Rcar (use dep::some_module::Rcar) Rcar
        "#]],
    );
    check(
        r#"
//- /lib.rs crate:dep
pub struct Barc;
pub struct Rcar;
pub struct Rc;
pub const RC: () = ();
pub mod some_module {
    pub struct Bar;
    pub struct Rcar;
    pub struct Rc;
    pub const RC: () = ();
}

//- /main.rs crate:main deps:dep
fn main() {
    rc$0
}
"#,
        expect![[r#"
            ct RC (use dep::RC)                    ()
            st Rc (use dep::Rc)                    Rc
            st Rcar (use dep::Rcar)              Rcar
            ct RC (use dep::some_module::RC)       ()
            st Rc (use dep::some_module::Rc)       Rc
            st Rcar (use dep::some_module::Rcar) Rcar
        "#]],
    );
    check(
        r#"
//- /lib.rs crate:dep
pub struct Barc;
pub struct Rcar;
pub struct Rc;
pub const RC: () = ();
pub mod some_module {
    pub struct Bar;
    pub struct Rcar;
    pub struct Rc;
    pub const RC: () = ();
}

//- /main.rs crate:main deps:dep
fn main() {
    RC$0
}
"#,
        expect![[r#"
            ct RC (use dep::RC)              ()
            ct RC (use dep::some_module::RC) ()
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
            st ThirdStruct (use dep::some_module::ThirdStruct)                ThirdStruct
            st AfterThirdStruct (use dep::some_module::AfterThirdStruct) AfterThirdStruct
            st ThiiiiiirdStruct (use dep::some_module::ThiiiiiirdStruct) ThiiiiiirdStruct
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
    dep::test_mod::TestStruct::weird_function();$0
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
            ct SPECIAL_CONST (use dep::test_mod::TestTrait) u8
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
    test_struct.random_method();$0
}
"#,
    );
}

#[test]
fn trait_method_fuzzy_completion_aware_of_fundamental_boxes() {
    let fixture = r#"
//- /fundamental.rs crate:fundamental
#[lang = "owned_box"]
#[fundamental]
pub struct Box<T>(T);
//- /foo.rs crate:foo
pub trait TestTrait {
    fn some_method(&self);
}
//- /main.rs crate:main deps:foo,fundamental
struct TestStruct;

impl foo::TestTrait for fundamental::Box<TestStruct> {
    fn some_method(&self) {}
}

fn main() {
    let t = fundamental::Box(TestStruct);
    t.$0
}
"#;

    check(
        fixture,
        expect![[r#"
            me some_method() (use foo::TestTrait) fn(&self)
        "#]],
    );

    check_edit(
        "some_method",
        fixture,
        r#"
use foo::TestTrait;

struct TestStruct;

impl foo::TestTrait for fundamental::Box<TestStruct> {
    fn some_method(&self) {}
}

fn main() {
    let t = fundamental::Box(TestStruct);
    t.some_method();$0
}
"#,
    );
}

#[test]
fn trait_method_fuzzy_completion_aware_of_fundamental_references() {
    let fixture = r#"
//- /foo.rs crate:foo
pub trait TestTrait {
    fn some_method(&self);
}
//- /main.rs crate:main deps:foo
struct TestStruct;

impl foo::TestTrait for &TestStruct {
    fn some_method(&self) {}
}

fn main() {
    let t = &TestStruct;
    t.$0
}
"#;

    check(
        fixture,
        expect![[r#"
            me some_method() (use foo::TestTrait) fn(&self)
        "#]],
    );

    check_edit(
        "some_method",
        fixture,
        r#"
use foo::TestTrait;

struct TestStruct;

impl foo::TestTrait for &TestStruct {
    fn some_method(&self) {}
}

fn main() {
    let t = &TestStruct;
    t.some_method();$0
}
"#,
    );
}

#[test]
fn trait_completions_handle_associated_types() {
    let fixture = r#"
//- /foo.rs crate:foo
pub trait NotInScope {
    fn not_in_scope(&self);
}

pub trait Wrapper {
    type Inner: NotInScope;
    fn inner(&self) -> Self::Inner;
}

//- /main.rs crate:main deps:foo
use foo::Wrapper;

fn completion<T: Wrapper>(whatever: T) {
    whatever.inner().$0
}
"#;

    check(
        fixture,
        expect![[r#"
            me not_in_scope() (use foo::NotInScope) fn(&self)
        "#]],
    );

    check_edit(
        "not_in_scope",
        fixture,
        r#"
use foo::{NotInScope, Wrapper};

fn completion<T: Wrapper>(whatever: T) {
    whatever.inner().not_in_scope();$0
}
"#,
    );
}

#[test]
fn trait_method_fuzzy_completion_aware_of_unit_type() {
    let fixture = r#"
//- /test_trait.rs crate:test_trait
pub trait TestInto<T> {
    fn into(self) -> T;
}

//- /main.rs crate:main deps:test_trait
struct A;

impl test_trait::TestInto<A> for () {
    fn into(self) -> A {
        A
    }
}

fn main() {
    let a = ();
    a.$0
}
"#;

    check(
        fixture,
        expect![[r#"
            me into() (use test_trait::TestInto) fn(self) -> T
        "#]],
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
    dep::test_mod::TestAlias::random_method();$0
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
    dep::test_mod::TestStruct::another_function();$0
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
            ct SPECIAL_CONST (use dep::test_mod::TestTrait)           u8 DEPRECATED
            fn weird_function() (use dep::test_mod::TestTrait)      fn() DEPRECATED
            me random_method(…) (use dep::test_mod::TestTrait) fn(&self) DEPRECATED
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
fn config_prefer_absolute() {
    let fixture = r#"
//- /lib.rs crate:dep
pub mod foo {
    pub mod bar {
        pub struct Item;
    }
}

//- /main.rs crate:main deps:dep
use ::dep::foo::bar;

fn main() {
    Ite$0
}"#;
    let mut config = TEST_CONFIG;
    config.prefer_absolute = true;

    check_edit_with_config(
        config.clone(),
        "Item",
        fixture,
        r#"
use ::dep::foo::bar::{self, Item};

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
            st Item (use foo::bar) Item
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
            ct TEST_ASSOC (use foo::Item) usize
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
            ct TEST_ASSOC (use foo::bar) usize
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
            ct TEST_CONST (use foo::TEST_CONST) usize
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
            ct TEST_CONST (use foo::TEST_CONST)               usize
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
    Tes$0
}"#,
        expect![""],
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
            st Struct (use crate::Struct) Struct
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
fn flyimport_pattern_unstable_path() {
    check(
        r#"
//- /main.rs crate:main deps:std
fn function() {
    let foo$0
}
//- /std.rs crate:std
#[unstable]
pub mod unstable {
    pub struct FooStruct {}
}
"#,
        expect![""],
    );
    check(
        r#"
//- toolchain:nightly
//- /main.rs crate:main deps:std
fn function() {
    let foo$0
}
//- /std.rs crate:std
#[unstable]
pub mod unstable {
    pub struct FooStruct {}
}
"#,
        expect![[r#"
            st FooStruct (use std::unstable::FooStruct)
        "#]],
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
            st Barbara (use foo::Barbara) Barbara
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

#[test]
fn no_completions_for_external_doc_hidden_in_path() {
    check(
        r#"
//- /main.rs crate:main deps:dep
fn main() {
    Span$0
}
//- /lib.rs crate:dep
#[doc(hidden)]
pub mod bridge {
    pub mod server {
        pub trait Span
    }
}
pub mod bridge2 {
    #[doc(hidden)]
    pub mod server2 {
        pub trait Span
    }
}
"#,
        expect![""],
    );
    // unless re-exported
    check(
        r#"
//- /main.rs crate:main deps:dep
fn main() {
    Span$0
}
//- /lib.rs crate:dep
#[doc(hidden)]
pub mod bridge {
    pub mod server {
        pub trait Span
    }
}
pub use bridge::server::Span;
pub mod bridge2 {
    #[doc(hidden)]
    pub mod server2 {
        pub trait Span2
    }
}
pub use bridge2::server2::Span2;
"#,
        expect![[r#"
            tt Span (use dep::Span)
            tt Span2 (use dep::Span2)
        "#]],
    );
}

#[test]
fn flyimport_only_traits_in_impl_trait_block() {
    check(
        r#"
//- /main.rs crate:main deps:dep
pub struct Bar;

impl Foo$0 for Bar { }
//- /lib.rs crate:dep
pub trait FooTrait;

pub struct FooStruct;
"#,
        expect![[r#"
            tt FooTrait (use dep::FooTrait)
        "#]],
    );
}

#[test]
fn primitive_mod() {
    check(
        r#"
//- minicore: str
fn main() {
    str::from$0
}
"#,
        expect![[r#"
            fn from_utf8_unchecked(…) (use core::str) const unsafe fn(&[u8]) -> &str
        "#]],
    );
}

#[test]
fn trait_impl_on_slice_method_on_deref_slice_type() {
    check(
        r#"
//- minicore: deref, sized
struct SliceDeref;
impl core::ops::Deref for SliceDeref {
    type Target = [()];

    fn deref(&self) -> &Self::Target {
        &[]
    }
}
fn main() {
    SliceDeref.choose$0();
}
mod module {
    pub(super) trait SliceRandom {
        type Item;

        fn choose(&self);
    }

    impl<T> SliceRandom for [T] {
        type Item = T;

        fn choose(&self) {}
    }
}
"#,
        expect![[r#"
            me choose (use module::SliceRandom) fn(&self)
        "#]],
    );
}

#[test]
fn re_export_aliased() {
    check(
        r#"
mod outer {
    mod inner {
        pub struct BarStruct;
        pub fn bar_fun() {}
        pub mod bar {}
    }
    pub use inner::bar as foo;
    pub use inner::bar_fun as foo_fun;
    pub use inner::BarStruct as FooStruct;
}
fn function() {
    foo$0
}
"#,
        expect![[r#"
            st FooStruct (use outer::FooStruct) BarStruct
            md foo (use outer::foo)
            fn foo_fun() (use outer::foo_fun)        fn()
        "#]],
    );
}

#[test]
fn re_export_aliased_pattern() {
    check(
        r#"
mod outer {
    mod inner {
        pub struct BarStruct;
        pub fn bar_fun() {}
        pub mod bar {}
    }
    pub use inner::bar as foo;
    pub use inner::bar_fun as foo_fun;
    pub use inner::BarStruct as FooStruct;
}
fn function() {
    let foo$0
}
"#,
        expect![[r#"
            st FooStruct (use outer::FooStruct)
            md foo (use outer::foo)
        "#]],
    );
}

#[test]
fn intrinsics() {
    check(
        r#"
    //- /core.rs crate:core
    pub mod intrinsics {
        extern "rust-intrinsic" {
            pub fn transmute<Src, Dst>(src: Src) -> Dst;
        }
    }
    pub mod mem {
        pub use crate::intrinsics::transmute;
    }
    //- /main.rs crate:main deps:core
    fn function() {
            transmute$0
    }
"#,
        expect![[r#"
            fn transmute(…) (use core::mem::transmute) unsafe fn(Src) -> Dst
        "#]],
    );
    check(
        r#"
//- /core.rs crate:core
pub mod intrinsics {
    extern "rust-intrinsic" {
        pub fn transmute<Src, Dst>(src: Src) -> Dst;
    }
}
pub mod mem {
    pub use crate::intrinsics::transmute;
}
//- /main.rs crate:main deps:core
fn function() {
        mem::transmute$0
}
"#,
        expect![[r#"
            fn transmute(…) (use core::mem) unsafe fn(Src) -> Dst
        "#]],
    );
}

#[test]
fn excluded_trait_item_included_when_exact_match() {
    // FIXME: This does not work, we need to change the code.
    check_with_config(
        CompletionConfig {
            exclude_traits: &["ra_test_fixture::module2::ExcludedTrait".to_owned()],
            ..TEST_CONFIG
        },
        r#"
mod module2 {
    pub trait ExcludedTrait {
        fn foo(&self) {}
        fn bar(&self) {}
        fn baz(&self) {}
    }

    impl<T> ExcludedTrait for T {}
}

fn foo() {
    true.foo$0
}
        "#,
        expect![""],
    );
}

#[test]
fn excluded_via_attr() {
    check(
        r#"
mod module2 {
    #[rust_analyzer::completions(ignore_flyimport)]
    pub trait ExcludedTrait {
        fn foo(&self) {}
        fn bar(&self) {}
        fn baz(&self) {}
    }

    impl<T> ExcludedTrait for T {}
}

fn foo() {
    true.$0
}
        "#,
        expect![""],
    );
    check(
        r#"
mod module2 {
    #[rust_analyzer::completions(ignore_flyimport_methods)]
    pub trait ExcludedTrait {
        fn foo(&self) {}
        fn bar(&self) {}
        fn baz(&self) {}
    }

    impl<T> ExcludedTrait for T {}
}

fn foo() {
    true.$0
}
        "#,
        expect![""],
    );
    check(
        r#"
mod module2 {
    #[rust_analyzer::completions(ignore_methods)]
    pub trait ExcludedTrait {
        fn foo(&self) {}
        fn bar(&self) {}
        fn baz(&self) {}
    }

    impl<T> ExcludedTrait for T {}
}

fn foo() {
    true.$0
}
        "#,
        expect![""],
    );
    check(
        r#"
mod module2 {
    #[rust_analyzer::completions(ignore_flyimport)]
    pub trait ExcludedTrait {
        fn foo(&self) {}
        fn bar(&self) {}
        fn baz(&self) {}
    }

    impl<T> ExcludedTrait for T {}
}

fn foo() {
    ExcludedTrait$0
}
        "#,
        expect![""],
    );
    check(
        r#"
mod module2 {
    #[rust_analyzer::completions(ignore_methods)]
    pub trait ExcludedTrait {
        fn foo(&self) {}
        fn bar(&self) {}
        fn baz(&self) {}
    }

    impl<T> ExcludedTrait for T {}
}

fn foo() {
    ExcludedTrait$0
}
        "#,
        expect![[r#"
            tt ExcludedTrait (use module2::ExcludedTrait)
        "#]],
    );
    check(
        r#"
mod module2 {
    #[rust_analyzer::completions(ignore_flyimport)]
    pub struct Foo {}
}

fn foo() {
    Foo$0
}
        "#,
        expect![""],
    );
}
