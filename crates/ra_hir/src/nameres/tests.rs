use std::sync::Arc;

use salsa::Database;
use ra_db::{FilesDatabase, CrateGraph};
use relative_path::RelativePath;
use test_utils::assert_eq_text;

use crate::{
    self as hir,
    db::HirDatabase,
    mock::MockDatabase,
};

fn item_map(fixture: &str) -> (Arc<hir::ItemMap>, hir::ModuleId) {
    let (db, pos) = MockDatabase::with_position(fixture);
    let source_root = db.file_source_root(pos.file_id);
    let module = hir::source_binder::module_from_position(&db, pos)
        .unwrap()
        .unwrap();
    let module_id = module.def_id.loc(&db).module_id;
    (db.item_map(source_root).unwrap(), module_id)
}

fn check_module_item_map(map: &hir::ItemMap, module_id: hir::ModuleId, expected: &str) {
    let mut lines = map.per_module[&module_id]
        .items
        .iter()
        .map(|(name, res)| format!("{}: {}", name, dump_resolution(res)))
        .collect::<Vec<_>>();
    lines.sort();
    let actual = lines.join("\n");
    let expected = expected
        .trim()
        .lines()
        .map(|it| it.trim())
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq_text!(&expected, &actual);

    fn dump_resolution(resolution: &hir::Resolution) -> &'static str {
        match (
            resolution.def_id.types.is_some(),
            resolution.def_id.values.is_some(),
        ) {
            (true, true) => "t v",
            (true, false) => "t",
            (false, true) => "v",
            (false, false) => "_",
        }
    }
}

#[test]
fn item_map_smoke_test() {
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        mod foo;

        use crate::foo::bar::Baz;
        <|>

        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        pub struct Baz;
    ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Baz: t v
            foo: t
        ",
    );
}

#[test]
fn re_exports() {
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        mod foo;

        use self::foo::Baz;
        <|>

        //- /foo/mod.rs
        pub mod bar;

        pub use self::bar::Baz;

        //- /foo/bar.rs
        pub struct Baz;
    ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Baz: t v
            foo: t
        ",
    );
}

#[test]
fn item_map_contains_items_from_expansions() {
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        mod foo;

        use crate::foo::bar::Baz;
        <|>

        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        salsa::query_group! {
            trait Baz {}
        }
    ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Baz: t
            foo: t
        ",
    );
}

#[test]
fn item_map_using_self() {
    let (item_map, module_id) = item_map(
        "
            //- /lib.rs
            mod foo;
            use crate::foo::bar::Baz::{self};
            <|>
            //- /foo/mod.rs
            pub mod bar;
            //- /foo/bar.rs
            pub struct Baz;
        ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Baz: t v
            foo: t
        ",
    );
}

#[test]
fn item_map_across_crates() {
    let (mut db, sr) = MockDatabase::with_files(
        "
        //- /main.rs
        use test_crate::Baz;

        //- /lib.rs
        pub struct Baz;
    ",
    );
    let main_id = sr.files[RelativePath::new("/main.rs")];
    let lib_id = sr.files[RelativePath::new("/lib.rs")];

    let mut crate_graph = CrateGraph::default();
    let main_crate = crate_graph.add_crate_root(main_id);
    let lib_crate = crate_graph.add_crate_root(lib_id);
    crate_graph.add_dep(main_crate, "test_crate".into(), lib_crate);

    db.set_crate_graph(crate_graph);

    let source_root = db.file_source_root(main_id);
    let module = hir::source_binder::module_from_file_id(&db, main_id)
        .unwrap()
        .unwrap();
    let module_id = module.def_id.loc(&db).module_id;
    let item_map = db.item_map(source_root).unwrap();

    check_module_item_map(
        &item_map,
        module_id,
        "
            Baz: t v
            test_crate: t
        ",
    );
}

#[test]
fn typing_inside_a_function_should_not_invalidate_item_map() {
    let (mut db, pos) = MockDatabase::with_position(
        "
        //- /lib.rs
        mod foo;

        use crate::foo::bar::Baz;

        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        <|>
        salsa::query_group! {
            trait Baz {
                fn foo() -> i32 { 1 + 1 }
            }
        }
        ",
    );
    let source_root = db.file_source_root(pos.file_id);
    {
        let events = db.log_executed(|| {
            db.item_map(source_root).unwrap();
        });
        assert!(format!("{:?}", events).contains("item_map"))
    }

    let new_text = "
        salsa::query_group! {
            trait Baz {
                fn foo() -> i32 { 92 }
            }
        }
    "
    .to_string();

    db.query_mut(ra_db::FileTextQuery)
        .set(pos.file_id, Arc::new(new_text));

    {
        let events = db.log_executed(|| {
            db.item_map(source_root).unwrap();
        });
        assert!(
            !format!("{:?}", events).contains("item_map"),
            "{:#?}",
            events
        )
    }
}

#[test]
fn typing_inside_a_function_inside_a_macro_should_not_invalidate_item_map() {
    let (mut db, pos) = MockDatabase::with_position(
        "
        //- /lib.rs
        mod foo;<|>

        use crate::foo::bar::Baz;

        fn foo() -> i32 {
            1 + 1
        }
        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        pub struct Baz;
    ",
    );
    let source_root = db.file_source_root(pos.file_id);
    {
        let events = db.log_executed(|| {
            db.item_map(source_root).unwrap();
        });
        assert!(format!("{:?}", events).contains("item_map"))
    }

    let new_text = "
        mod foo;

        use crate::foo::bar::Baz;

        fn foo() -> i32 { 92 }
    "
    .to_string();

    db.query_mut(ra_db::FileTextQuery)
        .set(pos.file_id, Arc::new(new_text));

    {
        let events = db.log_executed(|| {
            db.item_map(source_root).unwrap();
        });
        assert!(
            !format!("{:?}", events).contains("item_map"),
            "{:#?}",
            events
        )
    }
}
