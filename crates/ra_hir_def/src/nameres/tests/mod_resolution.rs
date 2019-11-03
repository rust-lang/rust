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
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮Baz: _
    "###);
}

#[test]
fn nested_module_resolution() {
    let map = def_map(
        "
        //- /lib.rs
        mod n1;

        //- /n1.rs
        mod n2;

        //- /n1/n2.rs
        struct X;
        ",
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮n1: t
        ⋮
        ⋮crate::n1
        ⋮n2: t
        ⋮
        ⋮crate::n1::n2
        ⋮X: t v
    "###);
}

#[test]
fn module_resolution_works_for_non_standard_filenames() {
    let map = def_map(
        "
        //- /my_library.rs crate:my_library
        mod foo;
        use self::foo::Bar;

        //- /foo/mod.rs
        pub struct Bar;
        ",
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        "
        //- /lib.rs
        mod r#async;
        use self::r#async::Bar;

        //- /async.rs
        pub struct Bar;
        ",
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /lib.rs
        #[path = "bar/baz/foo.rs"]
        mod foo;
        use self::foo::Bar;

        //- /bar/baz/foo.rs
        pub struct Bar;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
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
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
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
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /main.rs
        #[path = "bar/baz/module.rs"]
        mod foo;
        pub struct Baz;

        //- /bar/baz/module.rs
        use super::Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /main.rs
        #[path = "module/mod.rs"]
        mod foo;

        //- /module/mod.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮Baz: t v
    "###);
}

#[test]
fn module_resolution_relative_path() {
    let map = def_map(
        r###"
        //- /main.rs
        mod foo;

        //- /foo.rs
        #[path = "./sub.rs"]
        pub mod foo_bar;

        //- /sub.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /main.rs
        mod foo;

        //- /foo/mod.rs
        #[path="../sub.rs"]
        pub mod foo_bar;

        //- /sub.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /main.rs
        #[path = "module/bar/mod.rs"]
        mod foo;

        //- /module/bar/mod.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮Baz: t v
    "###);
}

#[test]
fn module_resolution_explicit_path_mod_rs_with_win_separator() {
    let map = def_map(
        r###"
        //- /main.rs
        #[path = "module\bar\mod.rs"]
        mod foo;

        //- /module/bar/mod.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮foo: t
        ⋮
        ⋮crate::foo
        ⋮Baz: t v
    "###);
}

#[test]
fn module_resolution_decl_inside_inline_module_with_path_attribute() {
    let map = def_map(
        r###"
        //- /main.rs
        #[path = "models"]
        mod foo {
            mod bar;
        }

        //- /models/bar.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /main.rs
        mod foo {
            mod bar;
        }

        //- /foo/bar.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /main.rs
        #[path = "models/db"]
        mod foo {
            mod bar;
        }

        //- /models/db/bar.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
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
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /main.rs
        #[path = ""]
        mod foo {
            #[path = "users.rs"]
            mod bar;
        }

        //- /users.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /main.rs
        #[path = ""] // Should try to read `/` (a directory)
        mod foo;

        //- /foo.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
        ⋮crate
    "###);
}

#[test]
fn module_resolution_decl_inside_inline_module_relative_path() {
    let map = def_map(
        r###"
        //- /main.rs
        #[path = "./models"]
        mod foo {
            mod bar;
        }

        //- /models/bar.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
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
fn module_resolution_decl_inside_inline_module_in_mod_rs() {
    let map = def_map(
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
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
        r###"
        //- /main.rs
        mod foo;

        //- /foo.rs
        mod bar {
            #[path = "qwe.rs"]
            pub mod baz;
        }
        use self::bar::baz::Baz;

        //- /foo/bar/qwe.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
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
    let map = def_map(
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
    );

    assert_snapshot!(map, @r###"
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
    let db = TestDB::with_files(
        r"
        //- /lib.rs
        mod foo;
        mod bar;
        mod baz {}
        //- /foo.rs
        ",
    );
    let krate = db.crate_graph().iter().next().unwrap();

    let crate_def_map = db.crate_def_map(krate);

    insta::assert_debug_snapshot!(
        crate_def_map.diagnostics,
        @r###"
    [
        UnresolvedModule {
            module: CrateModuleId(
                0,
            ),
            declaration: AstId {
                file_id: HirFileId(
                    FileId(
                        FileId(
                            0,
                        ),
                    ),
                ),
                file_ast_id: FileAstId {
                    raw: ErasedFileAstId(
                        1,
                    ),
                    _ty: PhantomData,
                },
            },
            candidate: "bar.rs",
        },
    ]
    "###
    );
}

#[test]
fn module_resolution_decl_inside_module_in_non_crate_root_2() {
    let map = def_map(
        r###"
        //- /main.rs
        #[path="module/m2.rs"]
        mod module;

        //- /module/m2.rs
        pub mod submod;

        //- /module/submod.rs
        pub struct Baz;
        "###,
    );

    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮module: t
        ⋮
        ⋮crate::module
        ⋮submod: t
        ⋮
        ⋮crate::module::submod
        ⋮Baz: t v
    "###);
}

#[test]
fn nested_out_of_line_module() {
    let map = def_map(
        r###"
        //- /lib.rs
        mod a {
            mod b {
                mod c;
            }
        }

        //- /a/b/c.rs
        struct X;
        "###,
    );

    assert_snapshot!(map, @r###"
    crate
    a: t
    
    crate::a
    b: t
    
    crate::a::b
    c: t
    
    crate::a::b::c
    X: t v
    "###);
}

#[test]
fn nested_out_of_line_module_with_path() {
    let map = def_map(
        r###"
        //- /lib.rs
        mod a {
            #[path = "d/e"]
            mod b {
                mod c;
            }
        }

        //- /a/d/e/c.rs
        struct X;
        "###,
    );

    assert_snapshot!(map, @r###"
    crate
    a: t
    
    crate::a
    b: t
    
    crate::a::b
    c: t
    
    crate::a::b::c
    X: t v
    "###);
}
