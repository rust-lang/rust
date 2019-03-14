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
    assert_snapshot_matches!(map, @r###"
crate
nested: t
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
n1: t

crate::n1
n2: t

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
bar: t
Foo: t v
Bar: t v

crate::bar
bar: t
Foo: t v
Bar: t v
"###);
}
