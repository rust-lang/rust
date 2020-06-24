use super::*;

#[test]
fn glob_1() {
    let map = def_map(
        r"
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
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮Baz: t v
   ⋮Foo: t v
   ⋮bar: t
   ⋮foo: t
   ⋮
   ⋮crate::foo
   ⋮Baz: t v
   ⋮Foo: t v
   ⋮bar: t
   ⋮
   ⋮crate::foo::bar
   ⋮Baz: t v
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
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮Baz: t v
   ⋮Foo: t v
   ⋮bar: t
   ⋮foo: t
   ⋮
   ⋮crate::foo
   ⋮Baz: t v
   ⋮Foo: t v
   ⋮bar: t
   ⋮
   ⋮crate::foo::bar
   ⋮Baz: t v
   ⋮Foo: t v
   ⋮bar: t
    "###
    );
}

#[test]
fn glob_privacy_1() {
    let map = def_map(
        r"
        //- /lib.rs
        mod foo;
        use foo::*;

        //- /foo/mod.rs
        pub mod bar;
        pub use self::bar::*;
        struct PrivateStructFoo;

        //- /foo/bar.rs
        pub struct Baz;
        struct PrivateStructBar;
        pub use super::*;
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
        ⋮bar: t
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮Baz: t v
        ⋮PrivateStructFoo: t v
        ⋮bar: t
        ⋮
        ⋮crate::foo::bar
        ⋮Baz: t v
        ⋮PrivateStructBar: t v
        ⋮PrivateStructFoo: t v
        ⋮bar: t
    "###
    );
}

#[test]
fn glob_privacy_2() {
    let map = def_map(
        r"
        //- /lib.rs
        mod foo;
        use foo::*;
        use foo::bar::*;

        //- /foo/mod.rs
        mod bar;
        fn Foo() {};
        pub struct Foo {};

        //- /foo/bar.rs
        pub(super) struct PrivateBaz;
        struct PrivateBar;
        pub(crate) struct PubCrateStruct;
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Foo: t
        ⋮PubCrateStruct: t v
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮Foo: t v
        ⋮bar: t
        ⋮
        ⋮crate::foo::bar
        ⋮PrivateBar: t v
        ⋮PrivateBaz: t v
        ⋮PubCrateStruct: t v
    "###
    );
}

#[test]
fn glob_across_crates() {
    mark::check!(glob_across_crates);
    let map = def_map(
        r"
        //- /main.rs crate:main deps:test_crate
        use test_crate::*;

        //- /lib.rs crate:test_crate
        pub struct Baz;
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
    "###
    );
}

#[test]
fn glob_privacy_across_crates() {
    let map = def_map(
        r"
        //- /main.rs crate:main deps:test_crate
        use test_crate::*;

        //- /lib.rs crate:test_crate
        pub struct Baz;
        struct Foo;
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: t v
    "###
    );
}

#[test]
fn glob_enum() {
    mark::check!(glob_enum);
    let map = def_map(
        "
        //- /lib.rs
        enum Foo {
            Bar, Baz
        }
        use self::Foo::*;
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Bar: t v
        ⋮Baz: t v
        ⋮Foo: t
    "###
    );
}

#[test]
fn glob_enum_group() {
    mark::check!(glob_enum_group);
    let map = def_map(
        r"
        //- /lib.rs
        enum Foo {
            Bar, Baz
        }
        use self::Foo::{*};
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Bar: t v
        ⋮Baz: t v
        ⋮Foo: t
    "###
    );
}

#[test]
fn glob_shadowed_def() {
    let db = TestDB::with_files(
        r###"
        //- /lib.rs
        mod foo;
        mod bar;

        use foo::*;
        use bar::Baz;

        //- /foo.rs
        pub struct Baz;

        //- /bar.rs
        pub struct Baz;
        "###,
    );
    let krate = db.test_crate();

    let crate_def_map = db.crate_def_map(krate);
    let (_, root_module) = crate_def_map
        .modules
        .iter()
        .find(|(_, module_data)| module_data.parent.is_none())
        .expect("Root module not found");
    let visible_entries = root_module.scope.entries().collect::<Vec<_>>();
    insta::assert_debug_snapshot!(
        visible_entries,
        @r###"
    [
        (
            Name(
                Text(
                    "Baz",
                ),
            ),
            PerNs {
                types: Some(
                    (
                        AdtId(
                            StructId(
                                StructId(
                                    1,
                                ),
                            ),
                        ),
                        Module(
                            ModuleId {
                                krate: CrateId(
                                    0,
                                ),
                                local_id: Idx::<ModuleData>(0),
                            },
                        ),
                    ),
                ),
                values: Some(
                    (
                        AdtId(
                            StructId(
                                StructId(
                                    1,
                                ),
                            ),
                        ),
                        Module(
                            ModuleId {
                                krate: CrateId(
                                    0,
                                ),
                                local_id: Idx::<ModuleData>(0),
                            },
                        ),
                    ),
                ),
                macros: None,
            },
        ),
        (
            Name(
                Text(
                    "bar",
                ),
            ),
            PerNs {
                types: Some(
                    (
                        ModuleId(
                            ModuleId {
                                krate: CrateId(
                                    0,
                                ),
                                local_id: Idx::<ModuleData>(2),
                            },
                        ),
                        Module(
                            ModuleId {
                                krate: CrateId(
                                    0,
                                ),
                                local_id: Idx::<ModuleData>(0),
                            },
                        ),
                    ),
                ),
                values: None,
                macros: None,
            },
        ),
        (
            Name(
                Text(
                    "foo",
                ),
            ),
            PerNs {
                types: Some(
                    (
                        ModuleId(
                            ModuleId {
                                krate: CrateId(
                                    0,
                                ),
                                local_id: Idx::<ModuleData>(1),
                            },
                        ),
                        Module(
                            ModuleId {
                                krate: CrateId(
                                    0,
                                ),
                                local_id: Idx::<ModuleData>(0),
                            },
                        ),
                    ),
                ),
                values: None,
                macros: None,
            },
        ),
    ]
    "###
    );
}
