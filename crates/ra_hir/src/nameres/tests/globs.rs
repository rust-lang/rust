use super::*;

#[test]
fn glob_1() {
    let map = def_map(
        "
        //- /lib.rs
        mod foo;
        use foo::*;

        //- /foo/mod.rs
        pub mod bar;
        pub use self::bar::Baz;
        pub struct Foo;

        //- /foo/bar.rs
        pub struct Baz;
        ",
    );
    assert_snapshot_matches!(map, @r###"
crate
bar: t
Foo: t v
Baz: t v
foo: t

crate::foo
bar: t
Foo: t v
Baz: t v

crate::foo::bar
Baz: t v
"###
    );
}

#[test]
fn glob_2() {
    let map = def_map(
        "
        //- /lib.rs
        mod foo;
        use foo::*;

        //- /foo/mod.rs
        pub mod bar;
        pub use self::bar::*;
        pub struct Foo;

        //- /foo/bar.rs
        pub struct Baz;
        pub use super::*;
        ",
    );
    assert_snapshot_matches!(map, @r###"
crate
bar: t
Foo: t v
Baz: t v
foo: t

crate::foo
bar: t
Foo: t v
Baz: t v

crate::foo::bar
bar: t
Foo: t v
Baz: t v
"###
    );
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
