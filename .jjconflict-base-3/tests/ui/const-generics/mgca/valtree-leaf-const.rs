//@ compile-flags: -Znext-solver

#![feature(gca_macroless_args)]
#![feature(gca_const_items)]
#![feature(gca_min_const_items)]

const TUPLE: (&'static str, &'static str) = ("a", true);
//~^ ERROR mismatched type

fn main() {
    TUPLE;
}
