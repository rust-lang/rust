//@ aux-build:macro_def.rs

#![feature(rustc_attrs)]
#![allow(internal_features)]
#![deny(dead_code)]

#[macro_use]
extern crate macro_def;

#[derive(Clone)]
struct Struct(usize); //~ ERROR: field `0` is never read

macro_rules! foo {
    () => {
        #[rustc_trivial_field_reads]
        fn foo(s: &Struct) {
            println!("{}", s.0);
        }
    };
}

foo!();
bar!(Struct);

fn main() {
    let s = Struct(0);
    foo(&s);
    bar(&s);
}
