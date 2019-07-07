use super::*;

#[test]
fn name_res_works_for_broken_modules() {
    covers!(name_res_works_for_broken_modules);
    let map = def_map(
        "
        //- /lib.rs
        mod foo // no `;`, no body

        use self::foo::Baz;

        //- /foo/mod.rs
        pub mod bar;

        pub use self::bar::Baz;

        //- /foo/bar.rs
        pub struct Baz;
        ",
    );
    assert_snapshot_matches!(map, @r###"
        ⋮crate
        ⋮Baz: _
    "###);
}

#[test]
fn module_resolution_works_for_non_standard_filenames() {
    let map = def_map_with_crate_graph(
        "
        //- /my_library.rs
        mod foo;
        use self::foo::Bar;

        //- /foo/mod.rs
        pub struct Bar;
        ",
        crate_graph! {
            "my_library": ("/my_library.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
        ⋮Bar: t v
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮Bar: t v
    "###);
}

#[test]
fn module_resolution_works_for_raw_modules() {
    let map = def_map_with_crate_graph(
        "
        //- /library.rs
        mod r#async;
        use self::r#async::Bar;

        //- /async.rs
        pub struct Bar;
        ",
        crate_graph! {
            "library": ("/library.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
        ⋮Bar: t v
        ⋮async: t
        ⋮
        ⋮crate::async
        ⋮Bar: t v
    "###);
}

#[test]
fn module_resolution_decl_path() {
    let map = def_map_with_crate_graph(
        "
        //- /library.rs
        #[path = \"bar/baz/foo.rs\"]
        mod foo;
        use self::foo::Bar;

        //- /bar/baz/foo.rs
        pub struct Bar;
        ",
        crate_graph! {
            "library": ("/library.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
        ⋮Bar: t v
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮Bar: t v
    "###);
}

#[test]
fn module_resolution_module_with_path_in_mod_rs() {
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        mod foo;
        
        //- /foo/mod.rs
        #[path = \"baz.rs\"]
        pub mod bar;

        use self::bar::Baz;

        //- /foo/baz.rs
        pub struct Baz;
        ",
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
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
fn module_resolution_module_with_path_non_crate_root() {
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        mod foo;
        
        //- /foo.rs
        #[path = \"baz.rs\"]
        pub mod bar;

        use self::bar::Baz;

        //- /baz.rs
        pub struct Baz;
        ",
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
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
fn unresolved_module_diagnostics() {
    let diagnostics = MockDatabase::with_files(
        r"
        //- /lib.rs
        mod foo;
        mod bar;
        mod baz {}
        //- /foo.rs
        ",
    )
    .diagnostics();

    assert_snapshot_matches!(diagnostics, @r###"
"mod bar;": unresolved module
"###
    );
}
