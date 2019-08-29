use super::*;

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
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮Foo: t v
   ⋮nested: t
   ⋮structs: m
   ⋮
   ⋮crate::nested
   ⋮Bar: t v
   ⋮Baz: t v
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
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮m: m
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
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮Bar: t v
   ⋮Foo: t v
   ⋮bar: t
   ⋮
   ⋮crate::bar
   ⋮Bar: t v
   ⋮Foo: t v
   ⋮bar: t
    "###);
}

#[test]
fn unexpanded_macro_should_expand_by_fixedpoint_loop() {
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        macro_rules! baz {
            () => {
                use foo::bar;
            }
        }

        foo!();
        bar!();
        baz!();

        //- /lib.rs
        #[macro_export]
        macro_rules! foo {
            () => {
                struct Foo { field: u32 }
            }
        }
        #[macro_export]
        macro_rules! bar {
            () => {
                use foo::foo;
            }
        }
        ",
        crate_graph! {
            "main": ("/main.rs", ["foo"]),
            "foo": ("/lib.rs", []),
        },
    );
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮Foo: t v
   ⋮bar: m
   ⋮baz: m
   ⋮foo: m
    "###);
}

#[test]
fn macro_rules_from_other_crates_are_visible_with_macro_use() {
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        #[macro_use]
        extern crate foo;

        structs!(Foo, Bar)

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
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮Bar: t v
   ⋮Foo: t v
   ⋮bar: t
   ⋮foo: t
   ⋮structs: m
   ⋮
   ⋮crate::bar
   ⋮Bar: t v
   ⋮Foo: t v
   ⋮bar: t
   ⋮foo: t
   ⋮structs: m
    "###);
}
