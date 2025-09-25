//@ build-pass
//@ edition:2024
#![crate_type = "lib"]
#![feature(const_cmp)]
#![feature(const_trait_impl)]
#![feature(macro_metavar_expr)]
#![allow(incomplete_features)]
#![feature(macro_fragment_fields)]
#![feature(macro_fragments_more)]

macro_rules! assert_fn_name {
    ($name:literal, $f:fn) => {
        const _: () = {
            assert!(stringify!(${f.name}) == $name);
        };
    };
}

assert_fn_name! {
    "f1",
    fn f1() {}
}

assert_fn_name! {
    "f2",
    extern "C" fn f2() {}
}

macro_rules! assert_adt_name {
    ($name:literal, $a:adt) => {
        const _: () = {
            assert!(stringify!(${a.name}) == $name);
        };
    };
}

assert_adt_name! {
    "S",
    struct S;
}

assert_adt_name! {
    "S",
    pub(crate) struct S<T>(u32, T);
}

assert_adt_name! {
    "E",
    enum E {
        V1,
        V2,
    }
}

assert_adt_name! {
    "U",
    union U {
        f: f64,
        u: u64,
    }
}

macro_rules! assert_fn_vis {
    ($v:vis, $f:fn) => {
        const _: () = {
            assert!(stringify!(${f.vis}) == stringify!($v));
        };
    }
}

assert_fn_vis! {
    pub,
    pub fn f() {}
}

assert_fn_vis! {
    pub(crate),
    pub(crate) fn f() {}
}

assert_fn_vis! {
    ,
    fn f() {}
}

macro_rules! use_vis {
    ($f:fn) => {${f.vis} struct StructWithFnVis;}
}

mod module {
    use_vis! { pub fn f() {} }
}

const C: module::StructWithFnVis = module::StructWithFnVis;
