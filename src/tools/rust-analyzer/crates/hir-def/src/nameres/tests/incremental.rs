use base_db::{SourceDatabase, SourceDatabaseFileInputExt as _};
use test_fixture::WithFixture;

use crate::{db::DefDatabase, nameres::tests::TestDB, AdtId, ModuleDefId};

fn check_def_map_is_not_recomputed(ra_fixture_initial: &str, ra_fixture_change: &str) {
    let (mut db, pos) = TestDB::with_position(ra_fixture_initial);
    let krate = {
        let crate_graph = db.crate_graph();
        // Some of these tests use minicore/proc-macros which will be injected as the first crate
        crate_graph.iter().last().unwrap()
    };
    {
        let events = db.log_executed(|| {
            db.crate_def_map(krate);
        });
        assert!(format!("{events:?}").contains("crate_def_map"), "{events:#?}")
    }
    db.set_file_text(pos.file_id.file_id(), ra_fixture_change);

    {
        let events = db.log_executed(|| {
            db.crate_def_map(krate);
        });
        assert!(!format!("{events:?}").contains("crate_def_map"), "{events:#?}")
    }
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

#[test]
fn typing_inside_an_attribute_arg_should_not_invalidate_def_map() {
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
#[proc_macros::identity(foo)]
fn f() {}
",
    );
}
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

#[test]
fn typing_inside_a_derive_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- proc_macros: derive_identity
//- minicore:derive
//- /lib.rs
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
$0
#[derive(proc_macros::DeriveIdentity)]
#[allow()]
struct S;
",
        r"
#[derive(proc_macros::DeriveIdentity)]
#[allow(dead_code)]
struct S;
",
    );
}

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
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 4);
        });
        let n_recalculated_item_trees =
            events.iter().filter(|it| it.contains("item_tree(")).count();
        assert_eq!(n_recalculated_item_trees, 6);
        let n_reparsed_macros =
            events.iter().filter(|it| it.contains("parse_macro_expansion(")).count();
        assert_eq!(n_reparsed_macros, 3);
    }

    let new_text = r#"
m!(X);
fn quux() { 92 }
m!(Y);
m!(Z);
"#;
    db.set_file_text(pos.file_id.file_id(), new_text);

    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 4);
        });
        let n_recalculated_item_trees = events.iter().filter(|it| it.contains("item_tree")).count();
        assert_eq!(n_recalculated_item_trees, 1);
        let n_reparsed_macros =
            events.iter().filter(|it| it.contains("parse_macro_expansion(")).count();
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
        let n_calculated_item_trees = events.iter().filter(|it| it.contains("item_tree(")).count();
        assert_eq!(n_calculated_item_trees, 1);
        let n_parsed_files = events.iter().filter(|it| it.contains("parse(")).count();
        assert_eq!(n_parsed_files, 1);
    }

    // Delete the parse tree.
    base_db::ParseQuery.in_db(&db).purge();

    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 8);
            assert_eq!(module_data.scope.impls().count(), 1);

            for imp in module_data.scope.impls() {
                db.impl_data(imp);
            }

            for (_, res) in module_data.scope.resolutions() {
                match res.values.map(|(a, _, _)| a).or(res.types.map(|(a, _, _)| a)).unwrap() {
                    ModuleDefId::FunctionId(f) => _ = db.function_data(f),
                    ModuleDefId::AdtId(adt) => match adt {
                        AdtId::StructId(it) => _ = db.struct_data(it),
                        AdtId::UnionId(it) => _ = db.union_data(it),
                        AdtId::EnumId(it) => _ = db.enum_data(it),
                    },
                    ModuleDefId::ConstId(it) => _ = db.const_data(it),
                    ModuleDefId::StaticId(it) => _ = db.static_data(it),
                    ModuleDefId::TraitId(it) => _ = db.trait_data(it),
                    ModuleDefId::TraitAliasId(it) => _ = db.trait_alias_data(it),
                    ModuleDefId::TypeAliasId(it) => _ = db.type_alias_data(it),
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
