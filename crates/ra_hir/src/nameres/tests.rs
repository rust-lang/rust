use std::sync::Arc;

use ra_db::SourceDatabase;
use test_utils::{assert_eq_text, covers};

use crate::{
    ItemMap,
    PersistentHirDatabase,
    mock::MockDatabase,
    nameres::crate_def_map::ModuleId,
};
use super::Resolution;

fn item_map(fixture: &str) -> (Arc<ItemMap>, ModuleId) {
    let (db, pos) = MockDatabase::with_position(fixture);
    let module = crate::source_binder::module_from_position(&db, pos).unwrap();
    let krate = module.krate(&db).unwrap();
    let module_id = module.module_id;
    (db.item_map(krate), module_id)
}

fn check_module_item_map(map: &ItemMap, module_id: ModuleId, expected: &str) {
    let mut lines = map[module_id]
        .items
        .iter()
        .map(|(name, res)| format!("{}: {}", name, dump_resolution(res)))
        .collect::<Vec<_>>();
    lines.sort();
    let actual = lines.join("\n");
    let expected = expected.trim().lines().map(|it| it.trim()).collect::<Vec<_>>().join("\n");
    assert_eq_text!(&expected, &actual);

    fn dump_resolution(resolution: &Resolution) -> &'static str {
        match (resolution.def.types.is_some(), resolution.def.values.is_some()) {
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
fn use_as() {
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        mod foo;

        use crate::foo::Baz as Foo;
        <|>

        //- /foo/mod.rs
        pub struct Baz;
    ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Foo: t v
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
fn glob_1() {
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        mod foo;
        use foo::*;
        <|>

        //- /foo/mod.rs
        pub mod bar;
        pub use self::bar::Baz;
        pub struct Foo;

        //- /foo/bar.rs
        pub struct Baz;
    ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Baz: t v
            Foo: t v
            bar: t
            foo: t
        ",
    );
}

#[test]
fn glob_2() {
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        mod foo;
        use foo::*;
        <|>

        //- /foo/mod.rs
        pub mod bar;
        pub use self::bar::*;
        pub struct Foo;

        //- /foo/bar.rs
        pub struct Baz;
        pub use super::*;
    ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Baz: t v
            Foo: t v
            bar: t
            foo: t
        ",
    );
}

#[test]
fn glob_enum() {
    covers!(glob_enum);
    let (item_map, module_id) = item_map(
        "
        //- /lib.rs
        enum Foo {
            Bar, Baz
        }
        use self::Foo::*;
        <|>
    ",
    );
    check_module_item_map(
        &item_map,
        module_id,
        "
            Bar: t v
            Baz: t v
            Foo: t
        ",
    );
}

#[test]
fn glob_across_crates() {
    covers!(glob_across_crates);
    let mut db = MockDatabase::with_files(
        "
        //- /main.rs
        use test_crate::*;

        //- /lib.rs
        pub struct Baz;
        ",
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["test_crate"]),
        "test_crate": ("/lib.rs", []),
    });
    let main_id = db.file_id_of("/main.rs");

    let module = crate::source_binder::module_from_file_id(&db, main_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
            Baz: t v
        ",
    );
}

#[test]
fn edition_2015_imports() {
    let mut db = MockDatabase::with_files(
        "
        //- /main.rs
        mod foo;
        mod bar;

        //- /bar.rs
        struct Bar;

        //- /foo.rs
        use bar::Bar;
        use other_crate::FromLib;

        //- /lib.rs
        struct FromLib;
    ",
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", "2015", ["other_crate"]),
        "other_crate": ("/lib.rs", "2018", []),
    });
    let foo_id = db.file_id_of("/foo.rs");

    let module = crate::source_binder::module_from_file_id(&db, foo_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
            Bar: t v
            FromLib: t v
        ",
    );
}

#[test]
fn module_resolution_works_for_non_standard_filenames() {
    let mut db = MockDatabase::with_files(
        "
        //- /my_library.rs
        mod foo;
        use self::foo::Bar;

        //- /foo/mod.rs
        pub struct Bar;
        ",
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "my_library": ("/my_library.rs", []),
    });
    let file_id = db.file_id_of("/my_library.rs");

    let module = crate::source_binder::module_from_file_id(&db, file_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let module_id = module.module_id;
    let item_map = db.item_map(krate);
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
fn std_prelude() {
    covers!(std_prelude);
    let mut db = MockDatabase::with_files(
        "
        //- /main.rs
        use Foo::*;

        //- /lib.rs
        mod prelude;
        #[prelude_import]
        use prelude::*;

        //- /prelude.rs
        pub enum Foo { Bar, Baz };
    ",
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["test_crate"]),
        "test_crate": ("/lib.rs", []),
    });
    let main_id = db.file_id_of("/main.rs");

    let module = crate::source_binder::module_from_file_id(&db, main_id).unwrap();
    eprintln!("module = {:?}", module);
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
            Bar: t v
            Baz: t v
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
    let mut db = MockDatabase::with_files(
        "
        //- /main.rs
        use test_crate::Baz;

        //- /lib.rs
        pub struct Baz;
        ",
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["test_crate"]),
        "test_crate": ("/lib.rs", []),
    });
    let main_id = db.file_id_of("/main.rs");

    let module = crate::source_binder::module_from_file_id(&db, main_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
        Baz: t v
        ",
    );
}

#[test]
fn extern_crate_rename() {
    let mut db = MockDatabase::with_files(
        "
        //- /main.rs
        extern crate alloc as alloc_crate;

        mod alloc;
        mod sync;

        //- /sync.rs
        use alloc_crate::Arc;

        //- /lib.rs
        struct Arc;
        ",
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["alloc"]),
        "alloc": ("/lib.rs", []),
    });
    let sync_id = db.file_id_of("/sync.rs");

    let module = crate::source_binder::module_from_file_id(&db, sync_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
        Arc: t v
        ",
    );
}

#[test]
fn extern_crate_rename_2015_edition() {
    let mut db = MockDatabase::with_files(
        "
        //- /main.rs
        extern crate alloc as alloc_crate;

        mod alloc;
        mod sync;

        //- /sync.rs
        use alloc_crate::Arc;

        //- /lib.rs
        struct Arc;
        ",
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", "2015", ["alloc"]),
        "alloc": ("/lib.rs", []),
    });
    let sync_id = db.file_id_of("/sync.rs");

    let module = crate::source_binder::module_from_file_id(&db, sync_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
        Arc: t v
        ",
    );
}

#[test]
fn import_across_source_roots() {
    let mut db = MockDatabase::with_files(
        "
        //- /lib.rs
        pub mod a {
            pub mod b {
                pub struct C;
            }
        }

        //- root /main/

        //- /main/main.rs
        use test_crate::a::b::C;
        ",
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main/main.rs", ["test_crate"]),
        "test_crate": ("/lib.rs", []),
    });
    let main_id = db.file_id_of("/main/main.rs");

    let module = crate::source_binder::module_from_file_id(&db, main_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
            C: t v
        ",
    );
}

#[test]
fn reexport_across_crates() {
    let mut db = MockDatabase::with_files(
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
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["test_crate"]),
        "test_crate": ("/lib.rs", []),
    });
    let main_id = db.file_id_of("/main.rs");

    let module = crate::source_binder::module_from_file_id(&db, main_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
        Baz: t v
        ",
    );
}

#[test]
fn values_dont_shadow_extern_crates() {
    let mut db = MockDatabase::with_files(
        "
        //- /main.rs
        fn foo() {}
        use foo::Bar;

        //- /foo/lib.rs
        pub struct Bar;
        ",
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["foo"]),
        "foo": ("/foo/lib.rs", []),
    });
    let main_id = db.file_id_of("/main.rs");

    let module = crate::source_binder::module_from_file_id(&db, main_id).unwrap();
    let krate = module.krate(&db).unwrap();
    let item_map = db.item_map(krate);

    check_module_item_map(
        &item_map,
        module.module_id,
        "
        Bar: t v
        foo: v
        ",
    );
}

fn check_item_map_is_not_recomputed(initial: &str, file_change: &str) {
    let (mut db, pos) = MockDatabase::with_position(initial);
    let module = crate::source_binder::module_from_file_id(&db, pos.file_id).unwrap();
    let krate = module.krate(&db).unwrap();
    {
        let events = db.log_executed(|| {
            db.item_map(krate);
        });
        assert!(format!("{:?}", events).contains("item_map"))
    }
    db.set_file_text(pos.file_id, Arc::new(file_change.to_string()));

    {
        let events = db.log_executed(|| {
            db.item_map(krate);
        });
        assert!(!format!("{:?}", events).contains("item_map"), "{:#?}", events)
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
