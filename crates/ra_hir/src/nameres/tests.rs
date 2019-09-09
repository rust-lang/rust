mod macros;
mod globs;
mod incremental;
mod primitives;
mod mod_resolution;

use std::sync::Arc;

use insta::assert_snapshot;
use ra_db::SourceDatabase;
use test_utils::covers;

use crate::{
    mock::{CrateGraphFixture, MockDatabase},
    Crate,
};

use super::*;

fn compute_crate_def_map(fixture: &str, graph: Option<CrateGraphFixture>) -> Arc<CrateDefMap> {
    let mut db = MockDatabase::with_files(fixture);
    if let Some(graph) = graph {
        db.set_crate_graph_from_fixture(graph);
    }
    let crate_id = db.crate_graph().iter().next().unwrap();
    let krate = Crate { crate_id };
    db.crate_def_map(krate)
}

fn render_crate_def_map(map: &CrateDefMap) -> String {
    let mut buf = String::new();
    go(&mut buf, map, "\ncrate", map.root);
    return buf.trim().to_string();

    fn go(buf: &mut String, map: &CrateDefMap, path: &str, module: CrateModuleId) {
        *buf += path;
        *buf += "\n";

        let mut entries = map.modules[module]
            .scope
            .items
            .iter()
            .map(|(name, res)| (name, res.def))
            .collect::<Vec<_>>();
        entries.sort_by_key(|(name, _)| *name);

        for (name, res) in entries {
            *buf += &format!("{}:", name);

            if res.types.is_some() {
                *buf += " t";
            }
            if res.values.is_some() {
                *buf += " v";
            }
            if res.macros.is_some() {
                *buf += " m";
            }
            if res.is_none() {
                *buf += " _";
            }

            *buf += "\n";
        }

        for (name, child) in map.modules[module].children.iter() {
            let path = path.to_string() + &format!("::{}", name);
            go(buf, map, &path, *child);
        }
    }
}

fn def_map(fixtute: &str) -> String {
    let dm = compute_crate_def_map(fixtute, None);
    render_crate_def_map(&dm)
}

fn def_map_with_crate_graph(fixture: &str, graph: CrateGraphFixture) -> String {
    let dm = compute_crate_def_map(fixture, Some(graph));
    render_crate_def_map(&dm)
}

#[test]
fn crate_def_map_smoke_test() {
    let map = def_map(
        "
        //- /lib.rs
        mod foo;
        struct S;
        use crate::foo::bar::E;
        use self::E::V;

        //- /foo/mod.rs
        pub mod bar;
        fn f() {}

        //- /foo/bar.rs
        pub struct Baz;
        enum E { V }
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮E: t
        ⋮S: t v
        ⋮V: t v
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮bar: t
        ⋮f: v
        ⋮
        ⋮crate::foo::bar
        ⋮Baz: t v
        ⋮E: t
    "###)
}

#[test]
fn bogus_paths() {
    covers!(bogus_paths);
    let map = def_map(
        "
        //- /lib.rs
        mod foo;
        struct S;
        use self;

        //- /foo/mod.rs
        use super;
        use crate;

        ",
    );
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮S: t v
   ⋮foo: t
   ⋮
   ⋮crate::foo
    "###
    )
}

#[test]
fn use_as() {
    let map = def_map(
        "
        //- /lib.rs
        mod foo;

        use crate::foo::Baz as Foo;

        //- /foo/mod.rs
        pub struct Baz;
        ",
    );
    assert_snapshot!(map,
        @r###"
   ⋮crate
   ⋮Foo: t v
   ⋮foo: t
   ⋮
   ⋮crate::foo
   ⋮Baz: t v
    "###
    );
}

#[test]
fn use_trees() {
    let map = def_map(
        "
        //- /lib.rs
        mod foo;

        use crate::foo::bar::{Baz, Quux};

        //- /foo/mod.rs
        pub mod bar;

        //- /foo/bar.rs
        pub struct Baz;
        pub enum Quux {};
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
        ⋮Quux: t
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮bar: t
        ⋮
        ⋮crate::foo::bar
        ⋮Baz: t v
        ⋮Quux: t
    "###);
}

#[test]
fn re_exports() {
    let map = def_map(
        "
        //- /lib.rs
        mod foo;

        use self::foo::Baz;

        //- /foo/mod.rs
        pub mod bar;

        pub use self::bar::Baz;

        //- /foo/bar.rs
        pub struct Baz;
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮Baz: t v
        ⋮bar: t
        ⋮
        ⋮crate::foo::bar
        ⋮Baz: t v
    "###);
}

#[test]
fn std_prelude() {
    covers!(std_prelude);
    let map = def_map_with_crate_graph(
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
        crate_graph! {
            "main": ("/main.rs", ["test_crate"]),
            "test_crate": ("/lib.rs", []),
        },
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Bar: t v
        ⋮Baz: t v
    "###);
}

#[test]
fn can_import_enum_variant() {
    covers!(can_import_enum_variant);
    let map = def_map(
        "
        //- /lib.rs
        enum E { V }
        use self::E::V;
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮E: t
        ⋮V: t v
    "###
    );
}

#[test]
fn edition_2015_imports() {
    let map = def_map_with_crate_graph(
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
        crate_graph! {
            "main": ("/main.rs", "2015", ["other_crate"]),
            "other_crate": ("/lib.rs", "2018", []),
        },
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮bar: t
        ⋮foo: t
        ⋮
        ⋮crate::bar
        ⋮Bar: t v
        ⋮
        ⋮crate::foo
        ⋮Bar: t v
        ⋮FromLib: t v
    "###);
}

#[test]
fn item_map_using_self() {
    let map = def_map(
        "
        //- /lib.rs
        mod foo;
        use crate::foo::bar::Baz::{self};
        //- /foo/mod.rs
        pub mod bar;
        //- /foo/bar.rs
        pub struct Baz;
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮bar: t
        ⋮
        ⋮crate::foo::bar
        ⋮Baz: t v
    "###);
}

#[test]
fn item_map_across_crates() {
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        use test_crate::Baz;

        //- /lib.rs
        pub struct Baz;
        ",
        crate_graph! {
            "main": ("/main.rs", ["test_crate"]),
            "test_crate": ("/lib.rs", []),
        },
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
    "###);
}

#[test]
fn extern_crate_rename() {
    let map = def_map_with_crate_graph(
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
        crate_graph! {
            "main": ("/main.rs", ["alloc"]),
            "alloc": ("/lib.rs", []),
        },
    );

    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮alloc_crate: t
   ⋮sync: t
   ⋮
   ⋮crate::sync
   ⋮Arc: t v
    "###);
}

#[test]
fn extern_crate_rename_2015_edition() {
    let map = def_map_with_crate_graph(
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
        crate_graph! {
            "main": ("/main.rs", "2015", ["alloc"]),
            "alloc": ("/lib.rs", []),
        },
    );

    assert_snapshot!(map,
        @r###"
   ⋮crate
   ⋮alloc_crate: t
   ⋮sync: t
   ⋮
   ⋮crate::sync
   ⋮Arc: t v
    "###
    );
}

#[test]
fn import_across_source_roots() {
    let map = def_map_with_crate_graph(
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
        crate_graph! {
            "main": ("/main/main.rs", ["test_crate"]),
            "test_crate": ("/lib.rs", []),
        },
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮C: t v
    "###);
}

#[test]
fn reexport_across_crates() {
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        use test_crate::Baz;

        //- /lib.rs
        pub use foo::Baz;

        mod foo;

        //- /foo.rs
        pub struct Baz;
        ",
        crate_graph! {
            "main": ("/main.rs", ["test_crate"]),
            "test_crate": ("/lib.rs", []),
        },
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
    "###);
}

#[test]
fn values_dont_shadow_extern_crates() {
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        fn foo() {}
        use foo::Bar;

        //- /foo/lib.rs
        pub struct Bar;
        ",
        crate_graph! {
            "main": ("/main.rs", ["foo"]),
            "foo": ("/foo/lib.rs", []),
        },
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Bar: t v
        ⋮foo: v
    "###);
}
