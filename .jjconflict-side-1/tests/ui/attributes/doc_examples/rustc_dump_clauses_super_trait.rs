//@ dont-require-annotations: ERROR
//@ compile-flags: --crate-type lib -Z ui-testing=no

#![feature(rustc_attrs)]

#[rustc_dump_clauses]
fn foo<T: Copy>(t: &T) -> T {
    *t
}
