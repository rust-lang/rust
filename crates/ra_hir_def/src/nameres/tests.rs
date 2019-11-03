mod globs;
mod incremental;
mod macros;
mod mod_resolution;
mod primitives;

use std::sync::Arc;

use insta::assert_snapshot;
use ra_db::{fixture::WithFixture, SourceDatabase};
use test_utils::covers;

use crate::{db::DefDatabase2, nameres::*, test_db::TestDB, CrateModuleId};

fn def_map(fixtute: &str) -> String {
    let dm = compute_crate_def_map(fixtute);
    render_crate_def_map(&dm)
}

fn compute_crate_def_map(fixture: &str) -> Arc<CrateDefMap> {
    let db = TestDB::with_files(fixture);
    let krate = db.crate_graph().iter().next().unwrap();
    db.crate_def_map(krate)
}

fn render_crate_def_map(map: &CrateDefMap) -> String {
    let mut buf = String::new();
    go(&mut buf, map, "\ncrate", map.root());
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
    let map = def_map(
        "
        //- /main.rs crate:main deps:test_crate
        use Foo::*;

        //- /lib.rs crate:test_crate
        mod prelude;
        #[prelude_import]
        use prelude::*;

        //- /prelude.rs
        pub enum Foo { Bar, Baz };
        ",
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
    let map = def_map(
        "
        //- /main.rs crate:main deps:other_crate edition:2015
        mod foo;
        mod bar;

        //- /bar.rs
        struct Bar;

        //- /foo.rs
        use bar::Bar;
        use other_crate::FromLib;

        //- /lib.rs crate:other_crate edition:2018
        struct FromLib;
        ",
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
    let map = def_map(
        "
        //- /main.rs crate:main deps:test_crate
        use test_crate::Baz;

        //- /lib.rs crate:test_crate
        pub struct Baz;
        ",
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
    "###);
}

#[test]
fn extern_crate_rename() {
    let map = def_map(
        "
        //- /main.rs crate:main deps:alloc
        extern crate alloc as alloc_crate;

        mod alloc;
        mod sync;

        //- /sync.rs
        use alloc_crate::Arc;

        //- /lib.rs crate:alloc
        struct Arc;
        ",
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
    let map = def_map(
        "
        //- /main.rs crate:main deps:alloc edition:2015
        extern crate alloc as alloc_crate;

        mod alloc;
        mod sync;

        //- /sync.rs
        use alloc_crate::Arc;

        //- /lib.rs crate:alloc
        struct Arc;
        ",
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
    let map = def_map(
        "
        //- /main.rs crate:main deps:test_crate
        use test_crate::a::b::C;

        //- root /test_crate/

        //- /test_crate/lib.rs crate:test_crate
        pub mod a {
            pub mod b {
                pub struct C;
            }
        }

        ",
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮C: t v
    "###);
}

#[test]
fn reexport_across_crates() {
    let map = def_map(
        "
        //- /main.rs crate:main deps:test_crate
        use test_crate::Baz;

        //- /lib.rs crate:test_crate
        pub use foo::Baz;

        mod foo;

        //- /foo.rs
        pub struct Baz;
        ",
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
    "###);
}

#[test]
fn values_dont_shadow_extern_crates() {
    let map = def_map(
        "
        //- /main.rs crate:main deps:foo
        fn foo() {}
        use foo::Bar;

        //- /foo/lib.rs crate:foo
        pub struct Bar;
        ",
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Bar: t v
        ⋮foo: v
    "###);
}

#[test]
fn cfg_not_test() {
    let map = def_map(
        r#"
        //- /main.rs crate:main deps:std
        use {Foo, Bar, Baz};

        //- /lib.rs crate:std
        #[prelude_import]
        pub use self::prelude::*;
        mod prelude {
            #[cfg(test)]
            pub struct Foo;
            #[cfg(not(test))]
            pub struct Bar;
            #[cfg(all(not(any()), feature = "foo", feature = "bar", opt = "42"))]
            pub struct Baz;
        }
        "#,
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Bar: t v
        ⋮Baz: _
        ⋮Foo: _
    "###);
}

#[test]
fn cfg_test() {
    let map = def_map(
        r#"
        //- /main.rs crate:main deps:std
        use {Foo, Bar, Baz};

        //- /lib.rs crate:std cfg:test,feature=foo,feature=bar,opt=42
        #[prelude_import]
        pub use self::prelude::*;
        mod prelude {
            #[cfg(test)]
            pub struct Foo;
            #[cfg(not(test))]
            pub struct Bar;
            #[cfg(all(not(any()), feature = "foo", feature = "bar", opt = "42"))]
            pub struct Baz;
        }
        "#,
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Bar: _
        ⋮Baz: t v
        ⋮Foo: t v
    "###);
}
