use base_db::{
    CrateDisplayName, CrateGraphBuilder, CrateName, CrateOrigin, CrateWorkspaceData,
    DependencyBuilder, Env, RootQueryDb, SourceDatabase,
};
use intern::Symbol;
use span::Edition;
use test_fixture::WithFixture;
use triomphe::Arc;

use crate::{
    AdtId, ModuleDefId,
    db::DefDatabase,
    nameres::{crate_def_map, tests::TestDB},
};

fn check_def_map_is_not_recomputed(
    #[rust_analyzer::rust_fixture] ra_fixture_initial: &str,
    #[rust_analyzer::rust_fixture] ra_fixture_change: &str,
) {
    let (mut db, pos) = TestDB::with_position(ra_fixture_initial);
    let krate = db.fetch_test_crate();
    {
        let events = db.log_executed(|| {
            crate_def_map(&db, krate);
        });
        assert!(
            format!("{events:?}").contains("crate_local_def_map"),
            "no crate def map computed:\n{events:#?}",
        )
    }
    db.set_file_text(pos.file_id.file_id(&db), ra_fixture_change);

    {
        let events = db.log_executed(|| {
            crate_def_map(&db, krate);
        });
        assert!(
            !format!("{events:?}").contains("crate_local_def_map"),
            "crate def map invalidated:\n{events:#?}",
        )
    }
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

    let events = db.log_executed(|| {
        for &krate in db.all_crates().iter() {
            crate_def_map(&db, krate);
        }
    });
    let invalidated_def_maps =
        events.iter().filter(|event| event.contains("crate_local_def_map")).count();
    assert_eq!(invalidated_def_maps, 1, "{events:#?}")
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
    {
        let events = db.log_executed(|| {
            let crate_def_map = crate_def_map(&db, krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 4);
        });
        let n_recalculated_item_trees =
            events.iter().filter(|it| it.contains("file_item_tree_shim")).count();
        assert_eq!(n_recalculated_item_trees, 6);
        let n_reparsed_macros =
            events.iter().filter(|it| it.contains("parse_macro_expansion_shim")).count();
        assert_eq!(n_reparsed_macros, 3);
    }

    let new_text = r#"
m!(X);
fn quux() { 92 }
m!(Y);
m!(Z);
"#;
    db.set_file_text(pos.file_id.file_id(&db), new_text);

    {
        let events = db.log_executed(|| {
            let crate_def_map = crate_def_map(&db, krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 4);
        });
        let n_recalculated_item_trees =
            events.iter().filter(|it| it.contains("file_item_tree_shim")).count();
        assert_eq!(n_recalculated_item_trees, 1, "{events:#?}");
        let n_reparsed_macros =
            events.iter().filter(|it| it.contains("parse_macro_expansion_shim")).count();
        assert_eq!(n_reparsed_macros, 0);
    }
}

#[test]
fn item_tree_prevents_reparsing() {
    // The `ItemTree` is used by both name resolution and the various queries in `adt.rs` and
    // `data.rs`. After computing the `ItemTree` and deleting the parse tree, we should be able to
    // run those other queries without triggering a reparse.

    let (db, pos) = TestDB::with_position(
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
    let krate = db.test_crate();
    {
        let events = db.log_executed(|| {
            db.file_item_tree(pos.file_id.into());
        });
        let n_calculated_item_trees =
            events.iter().filter(|it| it.contains("file_item_tree_shim")).count();
        assert_eq!(n_calculated_item_trees, 1);
        let n_parsed_files = events.iter().filter(|it| it.contains("parse")).count();
        assert_eq!(n_parsed_files, 1);
    }

    // FIXME(salsa-transition): bring this back
    // base_db::ParseQuery.in_db(&db).purge();

    {
        let events = db.log_executed(|| {
            let crate_def_map = crate_def_map(&db, krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 8);
            assert_eq!(module_data.scope.impls().count(), 1);

            for imp in module_data.scope.impls() {
                db.impl_signature(imp);
            }

            for (_, res) in module_data.scope.resolutions() {
                match res.values.map(|it| it.def).or(res.types.map(|it| it.def)).unwrap() {
                    ModuleDefId::FunctionId(f) => _ = db.function_signature(f),
                    ModuleDefId::AdtId(adt) => match adt {
                        AdtId::StructId(it) => _ = db.struct_signature(it),
                        AdtId::UnionId(it) => _ = db.union_signature(it),
                        AdtId::EnumId(it) => _ = db.enum_signature(it),
                    },
                    ModuleDefId::ConstId(it) => _ = db.const_signature(it),
                    ModuleDefId::StaticId(it) => _ = db.static_signature(it),
                    ModuleDefId::TraitId(it) => _ = db.trait_signature(it),
                    ModuleDefId::TraitAliasId(it) => _ = db.trait_alias_signature(it),
                    ModuleDefId::TypeAliasId(it) => _ = db.type_alias_signature(it),
                    ModuleDefId::EnumVariantId(_)
                    | ModuleDefId::ModuleId(_)
                    | ModuleDefId::MacroId(_)
                    | ModuleDefId::BuiltinType(_) => unreachable!(),
                }
            }
        });
        let n_reparsed_files = events.iter().filter(|it| it.contains("parse(")).count();
        assert_eq!(n_reparsed_files, 0);
    }
}
