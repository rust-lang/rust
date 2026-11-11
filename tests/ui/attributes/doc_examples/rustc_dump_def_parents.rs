//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

fn parent() {
    #[rustc_dump_def_parents]
    fn child() {}
}

struct Struct<const N: usize>;

const CONST: Struct<42> = Struct::<
    {
        #[rustc_dump_def_parents]
        fn baby() {}

        42
    },
>;
