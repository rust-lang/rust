// This is an equivalent of issue #50504, but for declarative macros.

#![feature(decl_macro, rustc_attrs)]

macro genmod($FromOutside: ident, $Outer: ident) {
    type A = $FromOutside;
    struct $Outer;
    mod inner {
        type A = $FromOutside; // `FromOutside` shouldn't be available from here
        type Inner = $Outer; // `Outer` shouldn't be available from here
    }
}

#[rustc_macro_transparency = "transparent"]
macro genmod_transparent() {
    type A = FromOutside;
    struct Outer;
    mod inner {
        type A = FromOutside; //~ ERROR cannot find type `FromOutside`
        type Inner = Outer; //~ ERROR cannot find type `Outer`
    }
}

macro_rules! genmod_legacy { () => {
    type A = FromOutside;
    struct Outer;
    mod inner {
        type A = FromOutside; //~ ERROR cannot find type `FromOutside`
        type Inner = Outer; //~ ERROR cannot find type `Outer`
    }
}}

fn check() {
    struct FromOutside;
    genmod!(FromOutside, Outer); //~ ERROR cannot find type `FromOutside`
                                 //~| ERROR cannot find type `Outer`
}

fn check_transparent() {
    struct FromOutside;
    genmod_transparent!();
}

fn check_legacy() {
    struct FromOutside;
    genmod_legacy!();
}

fn main() {}
