use base_db::{
    CrateDisplayName, CrateGraphBuilder, CrateName, CrateOrigin, CrateWorkspaceData,
    DependencyBuilder, Env, RootQueryDb, SourceDatabase,
};
use expect_test::{Expect, expect};
use intern::Symbol;
use span::Edition;
use test_fixture::WithFixture;
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    nameres::{crate_def_map, tests::TestDB},
};

fn check_def_map_is_not_recomputed(
    #[rust_analyzer::rust_fixture] ra_fixture_initial: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_change: &str,
    expecta: Expect,
    expectb: Expect,
) {
    let (mut db, pos) = TestDB::with_position(ra_fixture_initial);
    let krate = db.fetch_test_crate();
    execute_assert_events(
        &db,
        || {
            crate_def_map(&db, krate);
        },
        &[],
        expecta,
    );
    db.set_file_text(pos.file_id.file_id(&db), ra_fixture_change);

    execute_assert_events(
        &db,
        || {
            crate_def_map(&db, krate);
        },
        &[("crate_local_def_map", 0)],
        expectb,
    );
}

#[test]
fn crate_metadata_changes_should_not_invalidate_unrelated_def_maps() {
    let (mut db, files) = TestDB::with_many_files(
        r#"
//- /a.rs crate:a
pub fn foo() {}

//- /b.rs crate:b
pub struct Bar;

//- /c.rs crate:c deps:b
pub const BAZ: u32 = 0;
    "#,
    );

    for &krate in db.all_crates().iter() {
        crate_def_map(&db, krate);
    }

    let all_crates_before = db.all_crates();

    {
        // Add a dependency a -> b.
        let mut new_crate_graph = CrateGraphBuilder::default();

        let mut add_crate = |crate_name, root_file_idx: usize| {
            new_crate_graph.add_crate_root(
                files[root_file_idx].file_id(&db),
                Edition::CURRENT,
                Some(CrateDisplayName::from_canonical_name(crate_name)),
                None,
                Default::default(),
                None,
                Env::default(),
                CrateOrigin::Local { repo: None, name: Some(Symbol::intern(crate_name)) },
                false,
                Arc::new(
                    // FIXME: This is less than ideal
                    TryFrom::try_from(
                        &*std::env::current_dir().unwrap().as_path().to_string_lossy(),
                    )
                    .unwrap(),
                ),
                Arc::new(CrateWorkspaceData { data_layout: Err("".into()), toolchain: None }),
            )
        };
        let a = add_crate("a", 0);
        let b = add_crate("b", 1);
        let c = add_crate("c", 2);
        new_crate_graph
            .add_dep(c, DependencyBuilder::new(CrateName::new("b").unwrap(), b))
            .unwrap();
        new_crate_graph
            .add_dep(b, DependencyBuilder::new(CrateName::new("a").unwrap(), a))
            .unwrap();
        new_crate_graph.set_in_db(&mut db);
    }

    let all_crates_after = db.all_crates();
    assert!(
        Arc::ptr_eq(&all_crates_before, &all_crates_after),
        "the all_crates list should not have been invalidated"
    );
    execute_assert_events(
        &db,
        || {
            for &krate in db.all_crates().iter() {
                crate_def_map(&db, krate);
            }
        },
        &[("crate_local_def_map", 1)],
        expect![[r#"
            [
                "crate_local_def_map",
            ]
        "#]],
    );
}

#[test]
fn typing_inside_a_function_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- /lib.rs
mod foo;$0

use crate::foo::bar::Baz;

enum E { A, B }
use E::*;

fn foo() -> i32 {
    1 + 1
}

#[cfg(never)]
fn no() {}
//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
pub struct Baz;
",
        r"
mod foo;

use crate::foo::bar::Baz;

enum E { A, B }
use E::*;

fn foo() -> i32 { 92 }

#[cfg(never)]
fn no() {}
",
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "of_",
            ]
        "#]],
        expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "of_",
            ]
        "#]],
    );
}

#[test]
fn typing_inside_a_macro_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- /lib.rs
macro_rules! m {
    ($ident:ident) => {
        fn f() {
            $ident + $ident;
        };
    }
}
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
$0
m!(X);

pub struct S {}
",
        r"
m!(Y);

pub struct S {}
",
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "macro_def_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_macro_expansion_shim",
                "macro_arg_shim",
                "decl_macro_expander_shim",
            ]
        "#]],
        expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "macro_arg_shim",
                "parse_macro_expansion_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
            ]
        "#]],
    );
}

#[test]
fn typing_inside_an_attribute_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- proc_macros: identity
//- /lib.rs
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
$0
#[proc_macros::identity]
fn f() {}
",
        r"
#[proc_macros::identity]
fn f() { foo }
",
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "crate_local_def_map",
                "proc_macros_for_crate_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "macro_def_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_macro_expansion_shim",
                "expand_proc_macro_shim",
                "macro_arg_shim",
                "proc_macro_span_shim",
            ]
        "#]],
        expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "macro_arg_shim",
                "expand_proc_macro_shim",
                "parse_macro_expansion_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
            ]
        "#]],
    );
}

// Would be nice if this was the case, but as attribute inputs are stored in the item tree, this is
// not currently the case.
// #[test]
// fn typing_inside_an_attribute_arg_should_not_invalidate_def_map() {
//     check_def_map_is_not_recomputed(
//         r"
// //- proc_macros: identity
// //- /lib.rs
// mod foo;

// //- /foo/mod.rs
// pub mod bar;

// //- /foo/bar.rs
// $0
// #[proc_macros::identity]
// fn f() {}
// ",
//         r"
// #[proc_macros::identity(foo)]
// fn f() {}
// ",
//     );
// }

#[test]
fn typing_inside_macro_heavy_file_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- proc_macros: identity, derive_identity
//- /lib.rs
macro_rules! m {
    ($ident:ident) => {
        fn fm() {
            $ident + $ident;
        };
    }
}
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
$0
fn f() {}

m!(X);
macro_rules! m2 {
    ($ident:ident) => {
        fn f2() {
            $ident + $ident;
        };
    }
}
m2!(X);

#[proc_macros::identity]
#[derive(proc_macros::DeriveIdentity)]
pub struct S {}
",
        r"
fn f() {0}

m!(X);
macro_rules! m2 {
    ($ident:ident) => {
        fn f2() {
            $ident + $ident;
        };
    }
}
m2!(X);

#[proc_macros::identity]
#[derive(proc_macros::DeriveIdentity)]
pub struct S {}
",
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "macro_def_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_macro_expansion_shim",
                "macro_arg_shim",
                "decl_macro_expander_shim",
                "macro_def_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_macro_expansion_shim",
                "macro_arg_shim",
                "decl_macro_expander_shim",
                "crate_local_def_map",
                "proc_macros_for_crate_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "macro_def_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_macro_expansion_shim",
                "expand_proc_macro_shim",
                "macro_arg_shim",
                "proc_macro_span_shim",
            ]
        "#]],
        expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "macro_arg_shim",
                "macro_arg_shim",
                "decl_macro_expander_shim",
                "macro_arg_shim",
            ]
        "#]],
    );
}

// Would be nice if this was the case, but as attribute inputs are stored in the item tree, this is
// not currently the case.
// #[test]
// fn typing_inside_a_derive_should_not_invalidate_def_map() {
//     check_def_map_is_not_recomputed(
//         r"
// //- proc_macros: derive_identity
// //- minicore:derive
// //- /lib.rs
// mod foo;

// //- /foo/mod.rs
// pub mod bar;

// //- /foo/bar.rs
// $0
// #[derive(proc_macros::DeriveIdentity)]
// #[allow()]
// struct S;
// ",
//         r"
// #[derive(proc_macros::DeriveIdentity)]
// #[allow(dead_code)]
// struct S;
// ",
//     );
// }

#[test]
fn typing_inside_a_function_should_not_invalidate_item_expansions() {
    let (mut db, pos) = TestDB::with_position(
        r#"
//- /lib.rs
macro_rules! m {
    ($ident:ident) => {
        fn $ident() { };
    }
}
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
m!(X);
fn quux() { 1$0 }
m!(Y);
m!(Z);
"#,
    );
    let krate = db.test_crate();
    execute_assert_events(
        &db,
        || {
            let crate_def_map = crate_def_map(&db, krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 4);
        },
        &[("file_item_tree_query", 6), ("parse_macro_expansion_shim", 3)],
        expect![[r#"
            [
                "crate_local_def_map",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
                "macro_def_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_macro_expansion_shim",
                "macro_arg_shim",
                "decl_macro_expander_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_macro_expansion_shim",
                "macro_arg_shim",
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_macro_expansion_shim",
                "macro_arg_shim",
            ]
        "#]],
    );

    let new_text = r#"
m!(X);
fn quux() { 92 }
m!(Y);
m!(Z);
"#;
    db.set_file_text(pos.file_id.file_id(&db), new_text);

    execute_assert_events(
        &db,
        || {
            let crate_def_map = crate_def_map(&db, krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 4);
        },
        &[("file_item_tree_query", 1), ("parse_macro_expansion_shim", 0)],
        expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
                "macro_arg_shim",
                "macro_arg_shim",
                "macro_arg_shim",
            ]
        "#]],
    );
}

#[test]
fn item_tree_prevents_reparsing() {
    let (mut db, pos) = TestDB::with_position(
        r#"
pub struct S;
pub union U {}
pub enum E {
    Variant,
}
pub fn f(_: S) { $0 }
pub trait Tr {}
impl Tr for () {}
pub const C: u8 = 0;
pub static ST: u8 = 0;
pub type Ty = ();
"#,
    );

    execute_assert_events(
        &db,
        || {
            db.file_item_tree(pos.file_id.into());
        },
        &[("file_item_tree_query", 1), ("parse", 1)],
        expect![[r#"
            [
                "file_item_tree_query",
                "ast_id_map_shim",
                "parse_shim",
                "real_span_map_shim",
            ]
        "#]],
    );

    let file_id = pos.file_id.file_id(&db);
    let file_text = db.file_text(file_id).text(&db);
    db.set_file_text(file_id, &format!("{file_text}\n"));

    execute_assert_events(
        &db,
        || {
            db.file_item_tree(pos.file_id.into());
        },
        &[("file_item_tree_query", 1), ("parse", 1)],
        expect![[r#"
            [
                "parse_shim",
                "ast_id_map_shim",
                "file_item_tree_query",
                "real_span_map_shim",
            ]
        "#]],
    );
}

fn execute_assert_events(
    db: &TestDB,
    f: impl FnOnce(),
    required: &[(&str, usize)],
    expect: Expect,
) {
    let events = db.log_executed(f);
    for (event, count) in required {
        let n = events.iter().filter(|it| it.contains(event)).count();
        assert_eq!(n, *count, "Expected {event} to be executed {count} times, but only got {n}");
    }
    expect.assert_debug_eq(&events);
}
