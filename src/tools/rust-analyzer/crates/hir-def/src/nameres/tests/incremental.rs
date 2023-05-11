use std::sync::Arc;

use base_db::SourceDatabaseExt;

use crate::{AdtId, ModuleDefId};

use super::*;

fn check_def_map_is_not_recomputed(ra_fixture_initial: &str, ra_fixture_change: &str) {
    let (mut db, pos) = TestDB::with_position(ra_fixture_initial);
    let krate = db.test_crate();
    {
        let events = db.log_executed(|| {
            db.crate_def_map(krate);
        });
        assert!(format!("{events:?}").contains("crate_def_map"), "{events:#?}")
    }
    db.set_file_text(pos.file_id, Arc::new(ra_fixture_change.to_string()));

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
    let (mut db, pos) = TestDB::with_position(
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
        ",
    );
    let krate = db.test_crate();
    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 1);
        });
        assert!(format!("{events:?}").contains("crate_def_map"), "{events:#?}")
    }
    db.set_file_text(pos.file_id, Arc::new("m!(Y);".to_string()));

    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 1);
        });
        assert!(!format!("{events:?}").contains("crate_def_map"), "{events:#?}")
    }
}

#[test]
fn typing_inside_a_function_should_not_invalidate_expansions() {
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
        let n_recalculated_item_trees = events.iter().filter(|it| it.contains("item_tree")).count();
        assert_eq!(n_recalculated_item_trees, 6);
        let n_reparsed_macros =
            events.iter().filter(|it| it.contains("parse_macro_expansion")).count();
        assert_eq!(n_reparsed_macros, 3);
    }

    let new_text = r#"
m!(X);
fn quux() { 92 }
m!(Y);
m!(Z);
"#;
    db.set_file_text(pos.file_id, Arc::new(new_text.to_string()));

    {
        let events = db.log_executed(|| {
            let crate_def_map = db.crate_def_map(krate);
            let (_, module_data) = crate_def_map.modules.iter().last().unwrap();
            assert_eq!(module_data.scope.resolutions().count(), 4);
        });
        let n_recalculated_item_trees = events.iter().filter(|it| it.contains("item_tree")).count();
        assert_eq!(n_recalculated_item_trees, 1);
        let n_reparsed_macros =
            events.iter().filter(|it| it.contains("parse_macro_expansion")).count();
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
        let n_calculated_item_trees = events.iter().filter(|it| it.contains("item_tree")).count();
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
                match res.values.or(res.types).unwrap().0 {
                    ModuleDefId::FunctionId(f) => drop(db.function_data(f)),
                    ModuleDefId::AdtId(adt) => match adt {
                        AdtId::StructId(it) => drop(db.struct_data(it)),
                        AdtId::UnionId(it) => drop(db.union_data(it)),
                        AdtId::EnumId(it) => drop(db.enum_data(it)),
                    },
                    ModuleDefId::ConstId(it) => drop(db.const_data(it)),
                    ModuleDefId::StaticId(it) => drop(db.static_data(it)),
                    ModuleDefId::TraitId(it) => drop(db.trait_data(it)),
                    ModuleDefId::TraitAliasId(it) => drop(db.trait_alias_data(it)),
                    ModuleDefId::TypeAliasId(it) => drop(db.type_alias_data(it)),
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
