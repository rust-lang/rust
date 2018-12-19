use std::sync::Arc;

use salsa::Database;
use ra_db::{FilesDatabase, CrateGraph};
use ra_syntax::SmolStr;
use relative_path::RelativePath;

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
    let module_id = module.module_id;
    (db.item_map(source_root).unwrap(), module_id)
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
    let name = SmolStr::from("Baz");
    let resolution = &item_map.per_module[&module_id].items[&name];
    assert!(resolution.def_id.is_some());
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
    let module_id = module.module_id;
    let item_map = db.item_map(source_root).unwrap();

    let name = SmolStr::from("Baz");
    let resolution = &item_map.per_module[&module_id].items[&name];
    assert!(resolution.def_id.is_some());
}

#[test]
fn typing_inside_a_function_should_not_invalidate_item_map() {
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
            !format!("{:?}", events).contains("_item_map"),
            "{:#?}",
            events
        )
    }
}
