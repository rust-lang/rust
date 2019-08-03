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
        r###"
        //- /library.rs
        #[path = "bar/baz/foo.rs"]
        mod foo;
        use self::foo::Bar;

        //- /bar/baz/foo.rs
        pub struct Bar;
        "###,
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
        r###"
        //- /main.rs
        mod foo;
        
        //- /foo/mod.rs
        #[path = "baz.rs"]
        pub mod bar;

        use self::bar::Baz;

        //- /foo/baz.rs
        pub struct Baz;
        "###,
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
        r###"
        //- /main.rs
        mod foo;
        
        //- /foo.rs
        #[path = "baz.rs"]
        pub mod bar;

        use self::bar::Baz;

        //- /baz.rs
        pub struct Baz;
        "###,
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
fn module_resolution_module_decl_path_super() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = "bar/baz/module.rs"]
        mod foo;
        pub struct Baz;

        //- /bar/baz/module.rs
        use super::Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
        ⋮Baz: t v
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮Baz: t v
    "###);
}

#[test]
fn module_resolution_explicit_path_mod_rs() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = "module/mod.rs"]
        mod foo;

        //- /module/mod.rs
        pub struct Baz;
        "###,
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
    "###);
}

#[test]
fn module_resolution_relative_path() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        mod foo;

        //- /foo.rs
        #[path = "./sub.rs"]
        pub mod foo_bar;

        //- /sub.rs
        pub struct Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮foo_bar: t
        ⋮
        ⋮crate::foo::foo_bar
        ⋮Baz: t v
    "###);
}

#[test]
fn module_resolution_relative_path_2() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        mod foo;

        //- /foo/mod.rs
        #[path="../sub.rs"]
        pub mod foo_bar;

        //- /sub.rs
        pub struct Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮foo_bar: t
        ⋮
        ⋮crate::foo::foo_bar
        ⋮Baz: t v
    "###);
}

#[test]
fn module_resolution_explicit_path_mod_rs_2() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = "module/bar/mod.rs"]
        mod foo;

        //- /module/bar/mod.rs
        pub struct Baz;
        "###,
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
    "###);
}

#[test]
fn module_resolution_explicit_path_mod_rs_with_win_separator() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = "module\bar\mod.rs"]
        mod foo;

        //- /module/bar/mod.rs
        pub struct Baz;
        "###,
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
    "###);
}

#[test]
fn module_resolution_decl_inside_inline_module_with_path_attribute() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = "models"]
        mod foo {
            mod bar;
        }

        //- /models/bar.rs
        pub struct Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
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
fn module_resolution_decl_inside_inline_module() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        mod foo {
            mod bar;
        }

        //- /foo/bar.rs
        pub struct Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
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
fn module_resolution_decl_inside_inline_module_2_with_path_attribute() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = "models/db"]
        mod foo {
            mod bar;
        }

        //- /models/db/bar.rs
        pub struct Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
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
fn module_resolution_decl_inside_inline_module_3() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = "models/db"]
        mod foo {
            #[path = "users.rs"]
            mod bar;
        }

        //- /models/db/users.rs
        pub struct Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
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
fn module_resolution_decl_inside_inline_module_empty_path() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = ""]
        mod foo {
            #[path = "users.rs"]
            mod bar;
        }

        //- /foo/users.rs
        pub struct Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
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
fn module_resolution_decl_empty_path() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = ""]
        mod foo;

        //- /foo.rs
        pub struct Baz;
        "###,
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
    "###);
}

#[test]
fn module_resolution_decl_inside_inline_module_relative_path() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        #[path = "./models"]
        mod foo {
            mod bar;
        }

        //- /models/bar.rs
        pub struct Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
        ⋮crate
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
fn module_resolution_decl_inside_inline_module_in_crate_root() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        mod foo {
            #[path = "baz.rs"]
            mod bar;
        }
        use self::foo::bar::Baz;

        //- /foo/baz.rs
        pub struct Baz;
        "###,
        crate_graph! {
            "main": ("/main.rs", []),
        },
    );

    assert_snapshot_matches!(map, @r###"
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
fn module_resolution_decl_inside_inline_module_in_mod_rs() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        mod foo;

        //- /foo/mod.rs
        mod bar {
            #[path = "qwe.rs"]
            pub mod baz;
        }
        use self::bar::baz::Baz;

        //- /foo/bar/qwe.rs
        pub struct Baz;
        "###,
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
        ⋮baz: t
        ⋮
        ⋮crate::foo::bar::baz
        ⋮Baz: t v
    "###);
}

#[test]
fn module_resolution_decl_inside_inline_module_in_non_crate_root() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        mod foo;

        //- /foo.rs
        mod bar {
            #[path = "qwe.rs"]
            pub mod baz;
        }
        use self::bar::baz::Baz;

        //- /bar/qwe.rs
        pub struct Baz;
        "###,
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
        ⋮baz: t
        ⋮
        ⋮crate::foo::bar::baz
        ⋮Baz: t v
    "###);
}

#[test]
fn module_resolution_decl_inside_inline_module_in_non_crate_root_2() {
    let map = def_map_with_crate_graph(
        r###"
        //- /main.rs
        mod foo;

        //- /foo.rs
        #[path = "bar"]
        mod bar {
            pub mod baz;
        }
        use self::bar::baz::Baz;

        //- /bar/baz.rs
        pub struct Baz;
        "###,
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
        ⋮baz: t
        ⋮
        ⋮crate::foo::bar::baz
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
