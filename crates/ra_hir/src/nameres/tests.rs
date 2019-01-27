use std::sync::Arc;

use ra_db::{CrateGraph, SourceRootId, SourceDatabase};
use relative_path::RelativePath;
use test_utils::{assert_eq_text, covers};

use crate::{
    ItemMap, Resolution,
    db::HirDatabase,
    mock::MockDatabase,
    module_tree::ModuleId,
};

fn item_map(fixture: &str) -> (Arc<ItemMap>, ModuleId) {
    let (db, pos) = MockDatabase::with_position(fixture);
    let module = crate::source_binder::module_from_position(&db, pos).unwrap();
    let krate = module.krate(&db).unwrap();
    let module_id = module.module_id;
    (db.item_map(krate.crate_id), module_id)
}

fn item_map_custom_crate_root(fixture: &str, root: &str) -> (Arc<ItemMap>, ModuleId) {
    let (mut db, pos) = MockDatabase::with_position(fixture);

    let mut crate_graph = CrateGraph::default();
    crate_graph.add_crate_root(db.file_id(root));
    db.set_crate_graph(Arc::new(crate_graph));

    let module = crate::source_binder::module_from_position(&db, dbg!(pos)).unwrap();
    let krate = module.krate(&db).unwrap();
    let module_id = module.module_id;
    (db.item_map(krate.crate_id), module_id)
}

fn check_module_item_map(map: &ItemMap, module_id: ModuleId, expected: &str) {
    let mut lines = map[module_id]
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

    fn dump_resolution(resolution: &Resolution) -> &'static str {
        match (
            resolution.def.types.is_some(),
            resolution.def.values.is_some(),
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
fn use_trees() {
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        mod foo;

        use crate::foo::bar::{Baz, Quux};
        <|>

        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        pub struct Baz;
        pub enum Quux {};
    ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Baz: t v
            Quux: t
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
fn module_resolution_works_for_non_standard_filenames() {
    let (item_map, module_id) = item_map_custom_crate_root(
        "
        //- /my_library.rs
        mod foo;

        use self::foo::Bar;
        <|>

        //- /foo/mod.rs
        pub struct Bar;
    ",
        "/my_library.rs",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Bar: t v
            foo: t
        ",
    );
}

#[test]
fn name_res_works_for_broken_modules() {
    covers!(name_res_works_for_broken_modules);
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        mod foo // no `;`, no body

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
            Baz: _
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
fn item_map_enum_importing() {
    covers!(item_map_enum_importing);
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        enum E { V }
        use self::E::V;
        <|>
        ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
        E: t
        V: t v
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
    crate_graph
        .add_dep(main_crate, "test_crate".into(), lib_crate)
        .unwrap();

    db.set_crate_graph(Arc::new(crate_graph));

    let module = crate::source_binder::module_from_file_id(&db, main_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate.crate_id);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
            Baz: t v
            test_crate: t
        ",
    );
}

#[test]
fn import_across_source_roots() {
    let (mut db, sr) = MockDatabase::with_files(
        "
        //- /lib.rs
        pub mod a {
            pub mod b {
                pub struct C;
            }
        }
    ",
    );
    let lib_id = sr.files[RelativePath::new("/lib.rs")];

    let source_root = SourceRootId(1);

    let (sr2, pos) = db.add_fixture(
        source_root,
        "
        //- /main.rs
        use test_crate::a::b::C;
    ",
    );
    assert!(pos.is_none());

    let main_id = sr2.files[RelativePath::new("/main.rs")];

    eprintln!("lib = {:?}, main = {:?}", lib_id, main_id);

    let mut crate_graph = CrateGraph::default();
    let main_crate = crate_graph.add_crate_root(main_id);
    let lib_crate = crate_graph.add_crate_root(lib_id);
    crate_graph
        .add_dep(main_crate, "test_crate".into(), lib_crate)
        .unwrap();

    db.set_crate_graph(Arc::new(crate_graph));

    let module = crate::source_binder::module_from_file_id(&db, main_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate.crate_id);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
            C: t v
            test_crate: t
        ",
    );
}

#[test]
fn reexport_across_crates() {
    let (mut db, sr) = MockDatabase::with_files(
        "
        //- /main.rs
        use test_crate::Baz;

        //- /lib.rs
        pub use foo::Baz;

        mod foo;

        //- /foo.rs
        pub struct Baz;
    ",
    );
    let main_id = sr.files[RelativePath::new("/main.rs")];
    let lib_id = sr.files[RelativePath::new("/lib.rs")];

    let mut crate_graph = CrateGraph::default();
    let main_crate = crate_graph.add_crate_root(main_id);
    let lib_crate = crate_graph.add_crate_root(lib_id);
    crate_graph
        .add_dep(main_crate, "test_crate".into(), lib_crate)
        .unwrap();

    db.set_crate_graph(Arc::new(crate_graph));

    let module = crate::source_binder::module_from_file_id(&db, main_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate.crate_id);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
            Baz: t v
            test_crate: t
        ",
    );
}

fn check_item_map_is_not_recomputed(initial: &str, file_change: &str) {
    let (mut db, pos) = MockDatabase::with_position(initial);
    let module = crate::source_binder::module_from_file_id(&db, pos.file_id).unwrap();
    let krate = module.krate(&db).unwrap();
    {
        let events = db.log_executed(|| {
            db.item_map(krate.crate_id);
        });
        assert!(format!("{:?}", events).contains("item_map"))
    }
    db.set_file_text(pos.file_id, Arc::new(file_change.to_string()));

    {
        let events = db.log_executed(|| {
            db.item_map(krate.crate_id);
        });
        assert!(
            !format!("{:?}", events).contains("item_map"),
            "{:#?}",
            events
        )
    }
}

#[test]
fn typing_inside_a_function_should_not_invalidate_item_map() {
    check_item_map_is_not_recomputed(
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
        "
        mod foo;

        use crate::foo::bar::Baz;

        fn foo() -> i32 { 92 }
        ",
    );
}

#[test]
fn adding_inner_items_should_not_invalidate_item_map() {
    check_item_map_is_not_recomputed(
        "
        //- /lib.rs
        struct S { a: i32}
        enum E { A }
        trait T {
            fn a() {}
        }
        mod foo;<|>
        impl S {
            fn a() {}
        }
        use crate::foo::bar::Baz;
        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        pub struct Baz;
        ",
        "
        struct S { a: i32, b: () }
        enum E { A, B }
        trait T {
            fn a() {}
            fn b() {}
        }
        mod foo;<|>
        impl S {
            fn a() {}
            fn b() {}
        }
        use crate::foo::bar::Baz;
        ",
    );
}

#[test]
fn typing_inside_a_function_inside_a_macro_should_not_invalidate_item_map() {
    check_item_map_is_not_recomputed(
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
        "
        salsa::query_group! {
            trait Baz {
                fn foo() -> i32 { 92 }
            }
        }
        ",
    );
}
