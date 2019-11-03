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
fn glob_across_crates() {
    covers!(glob_across_crates);
    let map = def_map(
        "
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
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮Bar: t v
   ⋮Baz: t v
   ⋮Foo: t
    "###
    );
}
