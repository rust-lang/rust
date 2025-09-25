//@ edition:2024

#![crate_type = "lib"]
#![feature(macro_attr)]
#![allow(incomplete_features)]
#![feature(macro_fragments_more)]

macro_rules! parse_adt {
    ($a:adt) => {};
}

parse_adt! {
    struct S;
}

parse_adt! {
    pub(crate) struct S(u32);
}

parse_adt! {
    #[repr(C)]
    pub struct S {
        x: u64,
        y: u64,
    }
}

parse_adt! {
    enum E {}
}

parse_adt! {
    enum E {
        V1,
        V2,
    }
}

parse_adt! {
    enum E {
        V1,
        V2(u64),
        V3 { field: f64 },
    }
}

parse_adt! {
    union U {
        f: f64,
        u: u64,
    }
}

//~vv ERROR: expected a struct, enum, or union
parse_adt! {
    fn f() {}
}

//~vv ERROR: expected identifier
parse_adt! {
    struct;
}

//~vv ERROR: expected identifier
parse_adt! {
    enum;
}

//~vv ERROR: expected one of `!` or `::`, found `;`
parse_adt! {
    union;
}

macro_rules! adtattr {
    attr() ($a:adt) => { parse_adt!($a); };
}

#[adtattr]
struct S {
    u: u64,
    i: i64,
}

#[adtattr]
enum E1 {
    V1,
}

#[adtattr]
fn f() {}
//~^ ERROR: expected a struct, enum, or union

#[adtattr]
type T = u64;
//~^ ERROR: expected a struct, enum, or union

#[adtattr]
trait Trait {}
//~^ ERROR: expected a struct, enum, or union
