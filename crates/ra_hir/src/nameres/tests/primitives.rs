use super::*;

#[test]
fn primitive_reexport() {
    let map = def_map(
        "
        //- /lib.rs
        mod foo;
        use foo::int;

        //- /foo.rs
        pub use i32 as int;
        ",
    );
    assert_snapshot!(map, @r###"
   ⋮crate
   ⋮foo: t
   ⋮int: t
   ⋮
   ⋮crate::foo
   ⋮int: t
    "###
    );
}
