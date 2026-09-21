//@ compile-flags: -Znext-solver

#![feature(macroless_generic_const_args)]
#![feature(generic_const_args)]
#![feature(min_generic_const_args)]

const TUPLE: (&'static str, &'static str) = ("a", true);
//~^ ERROR mismatched type

fn main() {
    TUPLE;
}
