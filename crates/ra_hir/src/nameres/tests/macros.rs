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
    covers!(macro_rules_from_other_crates_are_visible_with_macro_use);
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        structs!(Foo);
        structs_priv!(Bar);
        structs_not_exported!(MacroNotResolved1);
        crate::structs!(MacroNotResolved2);

        mod bar;

        #[macro_use]
        extern crate foo;

        //- /bar.rs
        structs!(Baz);
        crate::structs!(MacroNotResolved3);

        //- /lib.rs
        #[macro_export]
        macro_rules! structs {
            ($i:ident) => { struct $i; }
        }

        macro_rules! structs_not_exported {
            ($i:ident) => { struct $i; }
        }

        mod priv_mod {
            #[macro_export]
            macro_rules! structs_priv {
                ($i:ident) => { struct $i; }
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
   ⋮
   ⋮crate::bar
   ⋮Baz: t v
    "###);
}

#[test]
fn prelude_is_macro_use() {
    covers!(prelude_is_macro_use);
    let map = def_map_with_crate_graph(
        "
        //- /main.rs
        structs!(Foo);
        structs_priv!(Bar);
        structs_outside!(Out);
        crate::structs!(MacroNotResolved2);

        mod bar;

        //- /bar.rs
        structs!(Baz);
        crate::structs!(MacroNotResolved3);

        //- /lib.rs
        #[prelude_import]
        use self::prelude::*;

        mod prelude {
            #[macro_export]
            macro_rules! structs {
                ($i:ident) => { struct $i; }
            }

            mod priv_mod {
                #[macro_export]
                macro_rules! structs_priv {
                    ($i:ident) => { struct $i; }
                }
            }
        }

        #[macro_export]
        macro_rules! structs_outside {
            ($i:ident) => { struct $i; }
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
   ⋮Out: t v
   ⋮bar: t
   ⋮
   ⋮crate::bar
   ⋮Baz: t v
    "###);
}

#[test]
fn prelude_cycle() {
    let map = def_map(
        "
        //- /lib.rs
        #[prelude_import]
        use self::prelude::*;

        declare_mod!();

        mod prelude {
            macro_rules! declare_mod {
                () => (mod foo {})
            }
        }
        ",
    );
    assert_snapshot!(map, @r###"
        ⋮crate
        ⋮prelude: t
        ⋮
        ⋮crate::prelude
        ⋮declare_mod: m
    "###);
}

#[test]
fn plain_macros_are_legacy_textual_scoped() {
    let map = def_map(
        r#"
        //- /main.rs
        mod m1;
        bar!(NotFoundNotMacroUse);

        mod m2 {
            foo!(NotFoundBeforeInside2);
        }

        macro_rules! foo {
            ($x:ident) => { struct $x; }
        }
        foo!(Ok);

        mod m3;
        foo!(OkShadowStop);
        bar!(NotFoundMacroUseStop);

        #[macro_use]
        mod m5 {
            #[macro_use]
            mod m6 {
                macro_rules! foo {
                    ($x:ident) => { fn $x() {} }
                }
            }
        }
        foo!(ok_double_macro_use_shadow);

        baz!(NotFoundBefore);
        #[macro_use]
        mod m7 {
            macro_rules! baz {
                ($x:ident) => { struct $x; }
            }
        }
        baz!(OkAfter);

        //- /m1.rs
        foo!(NotFoundBeforeInside1);
        macro_rules! bar {
            ($x:ident) => { struct $x; }
        }

        //- /m3/mod.rs
        foo!(OkAfterInside);
        macro_rules! foo {
            ($x:ident) => { fn $x() {} }
        }
        foo!(ok_shadow);

        #[macro_use]
        mod m4;
        bar!(OkMacroUse);

        //- /m3/m4.rs
        foo!(ok_shadow_deep);
        macro_rules! bar {
            ($x:ident) => { struct $x; }
        }
        "#,
    );
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮Ok: t v
   ⋮OkAfter: t v
   ⋮OkShadowStop: t v
   ⋮foo: m
   ⋮m1: t
   ⋮m2: t
   ⋮m3: t
   ⋮m5: t
   ⋮m7: t
   ⋮ok_double_macro_use_shadow: v
   ⋮
   ⋮crate::m7
   ⋮baz: m
   ⋮
   ⋮crate::m1
   ⋮bar: m
   ⋮
   ⋮crate::m5
   ⋮m6: t
   ⋮
   ⋮crate::m5::m6
   ⋮foo: m
   ⋮
   ⋮crate::m2
   ⋮
   ⋮crate::m3
   ⋮OkAfterInside: t v
   ⋮OkMacroUse: t v
   ⋮foo: m
   ⋮m4: t
   ⋮ok_shadow: v
   ⋮
   ⋮crate::m3::m4
   ⋮bar: m
   ⋮ok_shadow_deep: v
    "###);
}
