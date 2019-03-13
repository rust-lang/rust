use std::sync::Arc;

use ra_db::SourceDatabase;
use test_utils::covers;
use insta::assert_snapshot_matches;

use crate::{Crate, mock::{MockDatabase, CrateGraphFixture}, nameres::Resolution};

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
    return buf;

    fn go(buf: &mut String, map: &CrateDefMap, path: &str, module: ModuleId) {
        *buf += path;
        *buf += "\n";
        for (name, res) in map.modules[module].scope.items.iter() {
            *buf += &format!("{}: {}\n", name, dump_resolution(res))
        }
        for (name, child) in map.modules[module].children.iter() {
            let path = path.to_string() + &format!("::{}", name);
            go(buf, map, &path, *child);
        }
    }

    fn dump_resolution(resolution: &Resolution) -> &'static str {
        match (resolution.def.types.is_some(), resolution.def.values.is_some()) {
            (true, true) => "t v",
            (true, false) => "t",
            (false, true) => "v",
            (false, false) => "_",
        }
    }
}

fn def_map(fixtute: &str) -> String {
    let dm = compute_crate_def_map(fixtute, None);
    render_crate_def_map(&dm)
}

fn def_map_with_crate_graph(fixtute: &str, graph: CrateGraphFixture) -> String {
    let dm = compute_crate_def_map(fixtute, Some(graph));
    render_crate_def_map(&dm)
}

#[test]
fn crate_def_map_smoke_test() {
    let map = def_map(
        "
        //- /lib.rs
        mod foo;
        struct S;

        //- /foo/mod.rs
        pub mod bar;
        fn f() {}

        //- /foo/bar.rs
        pub struct Baz;
        enum E { V }
        ",
    );
    assert_snapshot_matches!(map, @r###"
crate
S: t v

crate::foo
f: v

crate::foo::bar
Baz: t v
E: t
"###
    )
}

#[test]
fn macro_rules_are_globally_visible() {
    let map = def_map(
        "
        //- /lib.rs
        macro_rules! structs {
            ($($i:ident),*) => {
                $(struct $i { field: u32 } )*
            }
        }
        structs!(Foo);
        mod nested;

        //- /nested.rs
        structs!(Bar, Baz);
        ",
    );
    assert_snapshot_matches!(map, @r###"
crate
Foo: t v

crate::nested
Bar: t v
Baz: t v
"###);
}

#[test]
fn macro_rules_can_define_modules() {
    let map = def_map(
        "
        //- /lib.rs
        macro_rules! m {
            ($name:ident) => { mod $name;  }
        }
        m!(n1);

        //- /n1.rs
        m!(n2)
        //- /n1/n2.rs
        struct X;
        ",
    );
    assert_snapshot_matches!(map, @r###"
crate

crate::n1

crate::n1::n2
X: t v
"###);
}

#[test]
fn macro_rules_from_other_crates_are_visible() {
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        foo::structs!(Foo, Bar)
        mod bar;

        //- /bar.rs
        use crate::*;

        //- /lib.rs
        #[macro_export]
        macro_rules! structs {
            ($($i:ident),*) => {
                $(struct $i { field: u32 } )*
            }
        }
        ",
        crate_graph! {
            "main": ("/main.rs", ["foo"]),
            "foo": ("/lib.rs", []),
        },
    );
    assert_snapshot_matches!(map, @r###"
crate
Foo: t v
Bar: t v

crate::bar
Foo: t v
Bar: t v
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
    assert_snapshot_matches!(map, @r###"
crate
Bar: t v
Baz: t v
"###);
}

#[test]
fn glob_across_crates() {
    covers!(glob_across_crates);
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        use test_crate::*;

        //- /lib.rs
        pub struct Baz;
        ",
        crate_graph! {
            "main": ("/main.rs", ["test_crate"]),
            "test_crate": ("/lib.rs", []),
        },
    );
    assert_snapshot_matches!(map, @r###"
crate
Baz: t v
"###
    );
}

#[test]
fn item_map_enum_importing() {
    covers!(item_map_enum_importing);
    let map = def_map(
        "
        //- /lib.rs
        enum E { V }
        use self::E::V;
        ",
    );
    assert_snapshot_matches!(map, @r###"
crate
V: t v
E: t
"###
    );
}

#[test]
fn glob_enum() {
    covers!(glob_enum);
    let map = def_map(
        "
        //- /lib.rs
        enum Foo {
            Bar, Baz
        }
        use self::Foo::*;
        ",
    );
    assert_snapshot_matches!(map, @r###"
crate
Foo: t
Bar: t v
Baz: t v
"###
    );
}
